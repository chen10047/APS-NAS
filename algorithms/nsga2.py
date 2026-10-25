
import numpy as np
from copy import deepcopy

# from algorithms import NSGAII
from .base import Algorithm as NSGAII
from model.individual import Individual
from model.program import CloneProgram
from model.functions import our_div, our_sqrt, our_log
from gplearn.functions import _function_map

DEFAULT_FUNCTION_SET = ['add', 'sub', 'mul', 'neg', 'div', 'log', 'sqrt']

def get_function_set(function_names):
    fs = []
    for name in function_names:
        if name == 'div':
            fs.append(our_div)
        elif name == 'log':
            fs.append(our_log)
        elif name == 'sqrt':
            fs.append(our_sqrt)
        else:
            fs.append(_function_map[name])
    return fs


class NSGAII(NSGAII):
    def __init__(self, **kwargs):
        self.pop_size = kwargs.get('pop_size', 100)
        self.n_generations = kwargs.get('n_generations', 50)  # 新增：总代数
        self.name = kwargs.get('name', 'NSGAII')
        # 遗传编程专用参数
        self.function_set_names = kwargs.get('function_set', DEFAULT_FUNCTION_SET)
        self.init_depth = kwargs.get('init_depth', (3,5))
        self.init_method = kwargs.get('init_method', 'half and half')
        self.const_range = kwargs.get('const_range', (-1., 1.))
        self.p_point_replace = kwargs.get('p_point_replace', 0.05)
        self.parsimony_coefficient = kwargs.get('parsimony_coefficient', 0.001)
        self.metric_to_dimension = kwargs.get('metric_to_dimension', {})
        self.required_dimensions = kwargs.get('required_dimensions', set())
        self.p_subtree_mutation = kwargs.get('p_subtree_mutation', 0.2)
        self.p_point_mutation = kwargs.get('p_point_mutation', 0.1)
        self.p_crossover = kwargs.get('p_crossover', 0.7)
        self.method_probs = np.cumsum([self.p_crossover,
                                       self.p_subtree_mutation,
                                       self.p_point_mutation])
        self.debug = kwargs.get('debug', False)
        self.early_stop_patience = kwargs.get('early_stop_patience', 0)  # 连续无改进的最大代数，0表示不启用
        self.early_stop_threshold = kwargs.get('early_stop_threshold', 0.001)  # Fused 提升的最小阈值

        self.sampling = None
        self.crossover = None
        self.mutation = None
        self.survival = None

        self._random_state = None
        self.pop = None
        self.problem = None
        self.seed = 0
        self.n_gen = 0
        self.n_eval = 0
        self.res_path = None

        from utils import ElitistArchive
        self.E_Archive_search = ElitistArchive(log_each_change=True)
        self.E_Archive_search_each_gen = []
        self.E_Archive_search_history = []
        self.nEvals_history = []
        self.nEvals_runningtime_each_gen = []
        self.executed_time_algorithm_history = [0.0]
        self.benchmark_time_algorithm_history = [0.0]
        self.indicator_time_history = [0.0]
        self.evaluated_time_history = [0.0]
        self.running_time_history = [0.0]
        self.tmp = 0.0
        self.max_depth = kwargs.get('max_depth', 8)  # 最大深度限制


    def set_hyperparameters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_up(self, problem, seed):
        self.problem = problem
        self.seed = seed
        np.random.seed(seed)
        import random
        random.seed(seed)
        self._setup()

    def solve(self, problem, seed):
        self.set_up(problem, seed)
        self._solve()

    def _solve(self):
        import time
        self.start_executed_time_algorithm = time.time()
        self.initialize()
        self.do_each_gen()

        # 早停相关变量
        best_fused = -float('inf')
        no_improve_counter = 0

        while self.n_gen < self.n_generations:
            self.n_gen += 1
            self.next(self.pop)
            self.do_each_gen()

            # ---------- 早停检查 ----------
            if self.early_stop_patience > 0:
                # 计算当前种群的最大 Fused（Fused = -F[0]）
                current_best_fused = max(-ind.F[0] for ind in self.pop)
                improvement = current_best_fused - best_fused
                if improvement > self.early_stop_threshold:
                    best_fused = current_best_fused
                    no_improve_counter = 0
                    if self.debug:
                        print(f"Fused improved to {best_fused:.6f}")
                else:
                    no_improve_counter += 1
                    if self.debug:
                        print(
                            f"No improvement for {no_improve_counter} generations (current best Fused={current_best_fused:.6f})")
                    if no_improve_counter >= self.early_stop_patience:
                        print(
                            f"Early stopping triggered: no Fused improvement for {self.early_stop_patience} generations.")
                        break
            # -----------------------------

        self.finalize()

    def do_each_gen(self):
        self._do_each_gen()


    def finalize(self):
        self._finalize()

    def reset(self):
        self.pop = None
        self.seed = 0
        self.n_gen = 0
        self.n_eval = 0
        self.res_path = None
        self.E_Archive_search = None
        self.start_executed_time_algorithm = 0.0
        self.finish_executed_time_algorithm = 0.0
        self.executed_time_algorithm_history = [0.0]
        self.benchmark_time_algorithm_history = [0.0]
        self.indicator_time_history = [0.0]
        self.evaluated_time_history = [0.0]
        self.running_time_history = [0.0]
        self.tmp = 0.0
        self.nEvals_history = []
        self.E_Archive_search_history = []
        self.nEvals_runningtime_each_gen = []
        self.E_Archive_search_each_gen = []

    def _setup(self):
        self.n_features = self.problem.X_train.shape[1]
        if hasattr(self.problem, 'feature_names'):
            self.feature_names = self.problem.feature_names
        else:
            self.feature_names = [f'X{i}' for i in range(self.n_features)]
        self.function_set = get_function_set(self.function_set_names)
        self.arities = {}
        for func in self.function_set:
            arity = func.arity
            self.arities.setdefault(arity, []).append(func)
        self._random_state = np.random.RandomState(self.seed)

    def _create_program(self, rng, program=None):
        """rng 必须是 np.random.RandomState 实例"""
        return CloneProgram(
            function_set=self.function_set,
            arities=self.arities,
            init_depth=self.init_depth,
            init_method=self.init_method,
            n_features=self.n_features,
            const_range=self.const_range,
            metric=None,
            p_point_replace=self.p_point_replace,
            parsimony_coefficient=self.parsimony_coefficient,
            random_state=rng,
            transformer=None,
            feature_names=self.feature_names,
            program=program,
            metric_to_dimension=self.metric_to_dimension,
            required_dimensions=self.required_dimensions
        )

    def _initialize(self):
        pop = []
        pop_exprs = set()  # 已生成的个体表达式，用于初始化去重
        for _ in range(self.pop_size):
            for attempt in range(1000):
                rng = np.random.RandomState(self._random_state.randint(2 ** 31))
                program = self._create_program(rng)
                expr_str = str(program)
                if (program.depth_ <= self.max_depth
                        and program._check_dimensions(program.program)
                        and expr_str not in pop_exprs):
                    F = self.evaluate(program)
                    ind = Individual()
                    ind.set('X', program)
                    ind.set('F', F)
                    pop.append(ind)
                    pop_exprs.add(expr_str)
                    break
            else:
                print("Warning: Could not generate a unique valid program, accepting anyway.")
                F = self.evaluate(program)
                ind = Individual()
                ind.set('X', program)
                ind.set('F', F)
                pop.append(ind)
                pop_exprs.add(str(program))
        self.pop = pop
        if self.survival is not None:
            _ = self.survival.do(self.pop, self.pop_size)


    def _evaluate(self, arch):
        F = self.problem.evaluate(arch)
        self.n_eval = self.problem.n_eval
        return F

    def _reproduce(self, parent_program):
        """深拷贝一个程序"""
        rng = np.random.RandomState(self._random_state.randint(2 ** 31))
        new_prog = self._create_program(rng, program=parent_program.program[:])
        new_prog.program = parent_program.program[:]
        return new_prog

    def _tournament_selection(self, parents, k=2):
        """从 parents 列表中通过 k 元锦标赛选择一个个体，处理 rank 为 None 的情况"""
        candidates = self._random_state.choice(len(parents), size=k, replace=False)
        best = candidates[0]
        for idx in candidates[1:]:
            rank_best = parents[best].get('rank')
            rank_idx = parents[idx].get('rank')
            # 处理 None 值（未排序时视为最差）
            if rank_best is None and rank_idx is None:
                # 都无 rank，随机选择（通过随机数）
                if self._random_state.rand() < 0.5:
                    best = idx
            elif rank_best is None:
                best = idx  # None 视为比有 rank 的差
            elif rank_idx is None:
                pass  # best 更好
            elif rank_idx < rank_best:
                best = idx
            elif rank_idx == rank_best:
                # 获取拥挤距离，注意 get 只接受一个参数，需要手动处理默认值
                crowd_best = parents[best].get('crowding')
                crowd_idx = parents[idx].get('crowding')
                if crowd_best is None:
                    crowd_best = -np.inf
                if crowd_idx is None:
                    crowd_idx = -np.inf
                if crowd_idx > crowd_best:
                    best = idx
        return parents[best].X

    def _select_parents(self, pop, current_gen, tournament_size=2):
        # 确保 rank 已设置
        if pop[0].get('rank') is None and self.survival is not None:
            _ = self.survival.do(pop, len(pop))

        # 构建父代池（保持与种群相同大小）
        parent_pool = []
        for _ in range(len(pop)):
            # 随机选择 tournament_size 个个体
            candidates = self._random_state.choice(len(pop), size=tournament_size, replace=False)
            # 选择 rank 最小的，若 rank 相同则选 crowding 最大的
            best = candidates[0]
            for idx in candidates[1:]:
                rank_best = pop[best].get('rank', np.inf)
                rank_idx = pop[idx].get('rank', np.inf)
                if rank_idx < rank_best:
                    best = idx
                elif rank_idx == rank_best:
                    crowd_best = pop[best].get('crowding', -np.inf)
                    crowd_idx = pop[idx].get('crowding', -np.inf)
                    if crowd_idx > crowd_best:
                        best = idx
            parent_pool.append(pop[best])
        return parent_pool

    def _mating(self, P):
        if P and P[0].get('rank') is None and self.survival is not None:
            _ = self.survival.do(P, len(P))

        # 根据当前代数选择父代池
        parent_pool = self._select_parents(P, self.n_gen)
        if len(parent_pool) == 0:
            print("Warning: No valid parents, reinitializing population...")
            self._initialize()
            return []

        offsprings = []
        n_offspring = self.pop_size
        rng = self._random_state
        max_crossover_retries = 20  # 交叉失败时的重试次数
        max_duplicate_retries = 100  # 去重重试次数

        # 收集当前种群所有表达式（而非仅抽样父代池），用于去重检查
        parent_exprs = {str(p.X) for p in P if p.X is not None}
        seen_offspring_exprs = set()  # 本次 mating 已生成的子代表达式

        while len(offsprings) < n_offspring:
            success = False
            duplicate_retry = 0
            while duplicate_retry < max_duplicate_retries and not success:
                # 选择两个不同的父代
                if len(parent_pool) == 1:
                    # 只有一个父代，只能繁殖
                    parent1 = parent_pool[0].X
                    parent2 = None
                else:
                    indices = rng.choice(len(parent_pool), size=2, replace=False)
                    parent1 = parent_pool[indices[0]].X
                    parent2 = parent_pool[indices[1]].X

                # 决定操作类型（根据概率）
                method = rng.rand()
                child = None
                operation_type = None

                # 交叉尝试（带重试）
                if method < self.method_probs[0] and parent2 is not None:
                    operation_type = 'crossover'
                    crossover_success = False
                    for _ in range(max_crossover_retries):
                        child1_prog, child2_prog, _, _ = parent1.our_crossover(parent2, rng)
                        if child1_prog is not None:
                            child_prog = child1_prog if rng.rand() < 0.5 else child2_prog
                            child_rng = np.random.RandomState(rng.randint(2 ** 31))
                            child = self._create_program(child_rng, program=child_prog)
                            child.program = child_prog
                            crossover_success = True
                            break
                        else:
                            # 交叉失败，重新选择父代
                            indices = rng.choice(len(parent_pool), size=2, replace=False)
                            parent1 = parent_pool[indices[0]].X
                            parent2 = parent_pool[indices[1]].X
                    if not crossover_success:
                        # 所有重试都失败，退化为点变异（对第一个父代）
                        operation_type = 'mutation'
                        new_prog, _ = parent1.our_point_mutation(rng)
                        if new_prog is not None:
                            child_rng = np.random.RandomState(rng.randint(2 ** 31))
                            child = self._create_program(child_rng, program=new_prog)
                            child.program = new_prog
                        else:
                            child = self._reproduce(parent1)
                elif method < self.method_probs[1]:
                    operation_type = 'mutation'
                    new_prog, _ = parent1.our_subtree_mutation(rng)
                    child_rng = np.random.RandomState(rng.randint(2 ** 31))
                    child = self._create_program(child_rng, program=new_prog)
                    child.program = new_prog
                elif method < self.method_probs[2]:
                    operation_type = 'mutation'
                    new_prog, _ = parent1.our_point_mutation(rng)
                    child_rng = np.random.RandomState(rng.randint(2 ** 31))
                    child = self._create_program(child_rng, program=new_prog)
                    child.program = new_prog
                else:
                    operation_type = 'reproduction'
                    print("reproduce")
                    child = self._reproduce(parent1)
                    # 为避免重复，增加变异概率
                    if rng.rand() < 0.5:
                        operation_type = 'mutation'
                        new_prog, _ = child.our_subtree_mutation(rng)
                        if new_prog is not None:
                            child_rng = np.random.RandomState(rng.randint(2 ** 31))
                            child = self._create_program(child_rng, program=new_prog)
                            child.program = new_prog

                # 评估子代
                if child is not None:
                    # ---------- 新增：检查维度覆盖 ----------
                    if not child._check_dimensions(child.program):
                        duplicate_retry += 1
                        continue
                    # -------------------------------------
                    expr_str = str(child)
                    # 检查是否与父代或已生成子代重复
                    if expr_str in parent_exprs or expr_str in seen_offspring_exprs:
                        duplicate_retry += 1
                        if operation_type == 'crossover':
                            continue
                        elif operation_type in ('mutation', 'reproduction'):
                            continue
                    else:
                        if child.depth_ > self.max_depth:
                            duplicate_retry += 1
                            continue
                        F = self.evaluate(child)
                        ind = Individual()
                        ind.set('X', child)
                        ind.set('F', F)
                        offsprings.append(ind)
                        seen_offspring_exprs.add(expr_str)
                        success = True
            if not success:
                print("Warning: Could not generate unique offspring, creating a brand new random program.")
                new_program = None
                for _ in range(20):
                    rng_new = np.random.RandomState(rng.randint(2 ** 31))
                    cand = self._create_program(rng_new)
                    cand_expr = str(cand)
                    if (cand._check_dimensions(cand.program)
                            and cand_expr not in parent_exprs
                            and cand_expr not in seen_offspring_exprs):
                        new_program = cand
                        break
                if new_program is None:
                    # 极端情况：20次仍无法生成唯一个体，接受最后一个候选
                    rng_new = np.random.RandomState(rng.randint(2 ** 31))
                    new_program = self._create_program(rng_new)
                    print("Warning: fallback offspring may duplicate an existing individual.")

                F = self.evaluate(new_program)
                ind = Individual()
                ind.set('X', new_program)
                ind.set('F', F)
                offsprings.append(ind)
                seen_offspring_exprs.add(str(new_program))

        return offsprings

    def _next(self, pop):
        if not pop:
            print("Population empty, reinitializing...")
            self._initialize()
            return
        offsprings = self._mating(pop)
        if not offsprings:
            print("No offspring generated, reinitializing population.")
            self._initialize()
            return
        pool = pop + offsprings
        new_pop = self.survival.do(pool, self.pop_size)
        self.pop = new_pop

    def _do_each_gen(self):
        if self.debug:
            print(f"\n{'-' * 60}")
            print(f"Generation {self.n_gen}  (Evals: {self.n_eval})")
            # print(f"{'=' * 60}")
            print(f"Population size: {len(self.pop)}")
            for i, ind in enumerate(self.pop):
                F = ind.get('F')
                if F is not None:
                    neg_fused, complexity_val = F
                    fused = -neg_fused
                else:
                    fused = complexity_val = None
                rank = ind.get('rank')
                expr = str(ind.get('X')) if ind.get('X') is not None else "None"
                # 截断过长表达式
                # print(f"Ind {i:2d}: rank={rank}, Fused={fused:.4f}, Complexity={complexity_val:.4f}")
                # 旧目标(leaf): print(f"Ind {i:2d}: rank={rank}, Fused={fused:.4f}, LeafCount={leaf_count_val}")
                # 旧目标(depth): print(f"Ind {i:2d}: rank={rank}, Fused={fused:.4f}, Depth={depth_val}")

            pareto_front = [ind for ind in self.pop if ind.get('rank') == 0]
            print(f"Pareto front size: {len(pareto_front)}")
            # print(f"{'=' * 60}\n")

    def _finalize(self):
        pass
