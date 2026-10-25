# program.py
from gplearn._program import _Program
from gplearn.functions import _Function
import numpy as np

# class CloneProgram(_Program):
#     def __init__(self,
#                  function_set,
#                  arities,
#                  init_depth,
#                  init_method,
#                  n_features,
#                  const_range,
#                  metric,
#                  p_point_replace,
#                  parsimony_coefficient,
#                  random_state,
#                  transformer=None,
#                  feature_names=None,
#                  program=None,
#                  metric_to_dimension=None,  # 新增：指标名称 -> 维度
#                  required_dimensions=None):  # 新增：必须覆盖的维度集合
#         self.feature_names = feature_names
#         # print(f"feature_names:{self.feature_names}")
#         self.metric_to_dimension = metric_to_dimension or {}
#         self.required_dimensions = set(required_dimensions) if required_dimensions else set()
#         self.required_dimensions_set = sorted(self.required_dimensions)
#         self.tmp_fitness = []
#         const_range = None
#         super().__init__(function_set, arities, init_depth, init_method,
#                          n_features, const_range, metric, p_point_replace,
#                          parsimony_coefficient, random_state, transformer,
#                          feature_names, program)
#         # print(const_range)

class CloneProgram(_Program):
    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 program=None,
                 metric_to_dimension=None,  # 新增：指标名称 -> 维度
                 required_dimensions=None): # 新增：必须覆盖的维度集合
        self.feature_names = feature_names
        self.metric_to_dimension = metric_to_dimension or {}
        self.required_dimensions = set(required_dimensions) if required_dimensions else set()
        self.required_dimensions_set = sorted(self.required_dimensions)
        self.tmp_fitness = []
        const_range = None

        # ================= 新增：构建允许的特征索引列表 =================
        self.allowed_feature_indices = []
        if self.required_dimensions:
            for i, name in enumerate(self.feature_names or []):
                dim = self.metric_to_dimension.get(name)
                if dim in self.required_dimensions:
                    self.allowed_feature_indices.append(i)
            # 若没有任何特征属于这些维度（配置错误），则回退到允许所有特征，避免崩溃
            if not self.allowed_feature_indices:
                print("Warning: No features found for required dimensions. Allowing all features.")
                self.allowed_feature_indices = list(range(self.n_features))
        else:
            # 未指定 required_dimensions 时，允许所有特征（保持原行为）
            self.allowed_feature_indices = list(range(self.n_features))
        # =============================================================

        super().__init__(function_set, arities, init_depth, init_method,
                         n_features, const_range, metric, p_point_replace,
                         parsimony_coefficient, random_state, transformer,
                         feature_names, program)

    def reproduce(self):
        import copy
        return copy.deepcopy(self)

    @property
    def leaf_count_(self):
        """叶子结点个数"""
        return sum(1 for node in self.program
                   if not callable(node) and not isinstance(node, _Function))

    def build_program(self, random_state, max_attempts=5000):
        """生成满足维度约束的表达树，最多尝试 max_attempts 次"""
        for attempt in range(max_attempts):
            program = self._build_program_once(random_state)
            # program = self._build_program_once()
            if self._check_dimensions(program):
                return program
        # 如果无法满足，打印警告并返回最后一次生成的程序
        print(f"失败，生成表达树已尝试： {max_attempts} ")
        return program

    def _build_program_once(self, random_state):
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # 选择一个二元函数作为根，避免退化程序
        binary_functions = [f for f in self.function_set if f.arity > 1]
        if not binary_functions:
            binary_functions = self.function_set
        function = random_state.choice(binary_functions)

        program = [function]
        terminal_stack = [function.arity]

        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(choice)
            if (depth < max_depth) and (method == 'full' or choice <= len(self.function_set)):
                function = random_state.choice(self.function_set)
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                # ========== 修改点：从允许的特征索引中随机选择 ==========
                if self.allowed_feature_indices:
                    terminal = int(random_state.choice(self.allowed_feature_indices))
                else:
                    terminal = int(random_state.randint(self.n_features))
                # ======================================================
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1
        return program

    # def _build_program_once(self, random_state):
    #     """原始构建逻辑（不做维度检查）"""
    #     if self.init_method == 'half and half':
    #         method = ('full' if random_state.randint(2) else 'grow')
    #     else:
    #         method = self.init_method
    #     max_depth = random_state.randint(*self.init_depth)
    #
    #     # 选择一个二元函数作为根，避免退化程序
    #     binary_functions = [f for f in self.function_set if f.arity > 1]
    #     if not binary_functions:
    #         binary_functions = self.function_set  # 如果没有二元函数，使用所有函数
    #     function = random_state.choice(binary_functions)
    #
    #     program = [function]
    #     terminal_stack = [function.arity]
    #
    #     while terminal_stack:
    #         depth = len(terminal_stack)
    #         # 选择函数或终端的概率
    #         choice = self.n_features + len(self.function_set)
    #         choice = random_state.randint(choice)
    #         if (depth < max_depth) and (method == 'full' or choice <= len(self.function_set)):
    #             # 添加函数
    #             function = random_state.choice(self.function_set)
    #             program.append(function)
    #             terminal_stack.append(function.arity)
    #         else:
    #             # 必须转成 Python int：numpy.int64 会被 gplearn 的 __str__ 当常数打印成 13.000
    #             terminal = int(random_state.randint(self.n_features))
    #
    #             # if self.const_range is not None:
    #             #     terminal = random_state.randint(self.n_features + 1)
    #             # else:
    #             #     terminal = random_state.randint(self.n_features)
    #             # if terminal == self.n_features:
    #             #     # 常数
    #             #     terminal = random_state.uniform(*self.const_range)
    #
    #             # print(f"[DEBUG] terminal type: {type(terminal)}, value: {terminal}")
    #             program.append(terminal)
    #             terminal_stack[-1] -= 1
    #             while terminal_stack[-1] == 0:
    #                 terminal_stack.pop()
    #                 if not terminal_stack:
    #                     return program
    #                 terminal_stack[-1] -= 1
    #     return program

    def _check_dimensions(self, program):
        """检查程序中的叶子节点是否覆盖了所有必需维度"""
        if not self.required_dimensions:
            return True
        # 收集所有叶子节点（非函数节点）
        leaves = [node for node in program if not callable(node) and not isinstance(node, _Function)]
        covered_dims = set()
        for leaf in leaves:
            if isinstance(leaf, int) and 0 <= leaf < self.n_features:
                # 特征索引 -> 指标名称 -> 维度
                if self.feature_names is not None and leaf < len(self.feature_names):
                    metric_name = self.feature_names[leaf]
                    dim = self.metric_to_dimension.get(metric_name)
                    if dim:
                        covered_dims.add(dim)
        return self.required_dimensions.issubset(covered_dims)

    # ----------------------------------------------
    def _parse_subtree(self, program, idx):
        """从索引 idx 开始解析一个子树，返回 (子树结束索引, 该子树覆盖的维度集合)"""
        node = program[idx]
        if callable(node) or isinstance(node, _Function):
            # 内部节点：函数
            arity = node.arity
            dims = set()
            next_idx = idx + 1
            for _ in range(arity):
                end, child_dims = self._parse_subtree(program, next_idx)
                dims.update(child_dims)
                next_idx = end
            return next_idx, dims
        else:
            # 叶子节点：特征索引或常数
            if isinstance(node, int) and 0 <= node < self.n_features:
                # 特征索引 -> 维度
                if self.feature_names and node < len(self.feature_names):
                    metric_name = self.feature_names[node]
                    dim = self.metric_to_dimension.get(metric_name)
                    return idx + 1, {dim} if dim else set()
                else:
                    return idx + 1, set()
            else:
                # 常数
                return idx + 1, set()

    # ---------- 生成指定维度集合的随机子树 ----------
    def _generate_subtree_with_dimensions(self, required_dims, random_state, max_depth=None):
        """生成一棵子树，其叶子节点覆盖的维度集合恰好等于 required_dims"""
        if max_depth is None:
            max_depth = random_state.randint(1, 3)  # 子树深度限制
        for _ in range(1000):
            sub_prog = self._build_random_subtree(random_state, max_depth)
            _, dims = self._parse_subtree(sub_prog, 0)
            if dims == required_dims:
                # print(f"dims:{required_dims}")
                return sub_prog
        # 失败时返回一个简单的叶子节点（使用 required_dims 中第一个维度的一个指标）
        # 找出 required_dims 中某个维度对应的任意特征索引
        sorted_dims = sorted(required_dims)
        for dim in sorted_dims:
            for i in range(self.n_features):
                if self.feature_names and i < len(self.feature_names):
                    if self.metric_to_dimension.get(self.feature_names[i]) == dim:
                        return [i]
        return [0]  # 保底

    # def _build_random_subtree(self, random_state, max_depth):
    #     """生成一个随机的子树（不检查维度）"""
    #     def build(depth):
    #         if depth >= max_depth or (random_state.rand() < 0.3 and depth > 0):
    #             # 生成叶子
    #             # 必须转成 Python int：numpy.int64 会被 gplearn 的 __str__ 当常数打印成 13.000
    #             return [int(random_state.randint(self.n_features))]
    #         else:
    #             # 生成函数
    #             func = random_state.choice(self.function_set)
    #             children = []
    #             for _ in range(func.arity):
    #                 children.extend(build(depth + 1))
    #             return [func] + children
    #
    #     return build(0)

    def _build_random_subtree(self, random_state, max_depth):
        """生成一个随机的子树（不检查维度），但叶子只能从允许的特征中选取"""

        def build(depth):
            if depth >= max_depth or (random_state.rand() < 0.3 and depth > 0):
                if self.allowed_feature_indices:
                    return [int(random_state.choice(self.allowed_feature_indices))]
                else:
                    return [int(random_state.randint(self.n_features))]
            else:
                func = random_state.choice(self.function_set)
                children = []
                for _ in range(func.arity):
                    children.extend(build(depth + 1))
                return [func] + children

        return build(0)

    # ------------- 点变异 ------------
    def our_point_mutation(self, random_state):
        """点变异：节点（特征）在同维度内替换，且只能替换为允许的特征"""
        new_program = self.program[:]
        node_idx = random_state.randint(len(new_program))
        node = new_program[node_idx]

        if callable(node) or isinstance(node, _Function):
            arity = node.arity
            candidates = [f for f in self.function_set if f.arity == arity and f != node]
            if candidates:
                new_node = random_state.choice(candidates)
                new_program[node_idx] = new_node
                return new_program, [node_idx]
        else:
            if isinstance(node, int) and 0 <= node < self.n_features:
                dim = None
                if self.feature_names and node < len(self.feature_names):
                    dim = self.metric_to_dimension.get(self.feature_names[node])

                if dim is not None:
                    # ========== 修改点：只从 allowed_feature_indices 中筛选同维度特征 ==========
                    same_dim_indices = []
                    for i in self.allowed_feature_indices:
                        if self.feature_names and i < len(self.feature_names):
                            if self.metric_to_dimension.get(self.feature_names[i]) == dim and i != node:
                                same_dim_indices.append(i)
                    # ======================================================================
                    if same_dim_indices:
                        new_node = int(random_state.choice(same_dim_indices))
                        new_program[node_idx] = new_node
                        return new_program, [node_idx]
        return new_program, []
    # def our_point_mutation(self, random_state):
    #     """点变异：节点（特征）在同维度内替换"""
    #     new_program = self.program[:]
    #     # 随机选择一个节点位置
    #     node_idx = random_state.randint(len(new_program))
    #     node = new_program[node_idx]
    #
    #     if callable(node) or isinstance(node, _Function):
    #         # 函数节点：替换为相同元数的其他函数
    #         arity = node.arity
    #         candidates = [f for f in self.function_set if f.arity == arity and f != node]
    #         if candidates:
    #             new_node = random_state.choice(candidates)
    #             new_program[node_idx] = new_node
    #             return new_program, [node_idx]
    #     else:
    #         # 叶子节点：如果是特征索引，在同维度内替换
    #         if isinstance(node, int) and 0 <= node < self.n_features:
    #             dim = None
    #             if self.feature_names and node < len(self.feature_names):
    #                 dim = self.metric_to_dimension.get(self.feature_names[node])
    #
    #             if dim is not None:
    #                 # 收集同维度的所有特征索引
    #                 same_dim_indices = []
    #                 for i in range(self.n_features):
    #
    #                     if self.feature_names and i < len(self.feature_names):
    #                         if self.metric_to_dimension.get(self.feature_names[i]) == dim and i != node:
    #                             same_dim_indices.append(i)
    #                 if same_dim_indices:
    #                     new_node = int(random_state.choice(same_dim_indices))
    #                     # new_node = random_state.choice(same_dim_indices)
    #                     new_program[node_idx] = new_node
    #                     return new_program, [node_idx]
    #     return new_program, []

    # ------------ 子树变异 ------------
    # def _generate_subtree_containing_dimensions(self, required_dims, random_state, max_depth=None):
    #     if max_depth is None:
    #         max_depth = random_state.randint(1, 4)
    #     max_attempts = 5000
    #     for _ in range(max_attempts):
    #         sub_prog = self._build_random_subtree(random_state, max_depth)
    #         _, dims = self._parse_subtree(sub_prog, 0)
    #         if required_dims.issubset(dims):
    #             return sub_prog
    #     return None

    # def _generate_subtree_containing_dimensions(self, required_dims, random_state, max_depth=None):
    #     """构造性地生成一棵子树，其叶子覆盖的维度集合包含 required_dims"""
    #     if not required_dims:
    #         # 没有维度要求，回退到随机生成（保留原随机逻辑）
    #         return self._build_random_subtree(random_state, max_depth or 2)
    #
    #     # 1. 为每个维度随机选择一个特征索引
    #     selected_leaves = []
    #     for dim in required_dims:
    #         candidates = []
    #         for i in range(self.n_features):
    #             if (self.feature_names and i < len(self.feature_names) and
    #                     self.metric_to_dimension.get(self.feature_names[i]) == dim):
    #                 candidates.append(i)
    #         if not candidates:
    #             # 理论上不会发生，但为了安全，回退到随机生成
    #             return self._build_random_subtree(random_state, max_depth or 2)
    #         # 必须转成 Python int：numpy.int64 会被 gplearn 的 __str__ 当常数打印成 13.000
    #         selected_leaves.append(int(random_state.choice(candidates)))
    #
    #     # 2. 如果只有一个叶子，直接返回
    #     if len(selected_leaves) == 1:
    #         return [selected_leaves[0]]
    #
    #     # 3. 组合成树
    #     return self._build_tree_from_leaves(selected_leaves, random_state)

    def _generate_subtree_containing_dimensions(self, required_dims, random_state, max_depth=None):
        """构造性地生成一棵子树，其叶子覆盖的维度集合包含 required_dims，且所有叶子来自允许的特征"""
        if not required_dims:
            return self._build_random_subtree(random_state, max_depth or 2)

        selected_leaves = []
        for dim in required_dims:
            candidates = [i for i in self.allowed_feature_indices
                          if (self.feature_names and i < len(self.feature_names) and
                              self.metric_to_dimension.get(self.feature_names[i]) == dim)]
            if not candidates:
                print("random generate subtree failed!!")
                return self._build_random_subtree(random_state, max_depth or 2)
            selected_leaves.append(int(random_state.choice(candidates)))

        if len(selected_leaves) == 1:
            return [selected_leaves[0]]

        return self._build_tree_from_leaves(selected_leaves, random_state)

    def _build_tree_from_leaves(self, leaves, random_state):
        """将叶子列表组合成一棵二叉树（前缀表达式）"""
        if len(leaves) == 1:
            return [leaves[0]]
        mid = len(leaves) // 2
        left_leaves = leaves[:mid]
        right_leaves = leaves[mid:]
        left_tree = self._build_tree_from_leaves(left_leaves, random_state)
        right_tree = self._build_tree_from_leaves(right_leaves, random_state)
        # 随机选择一个二元函数作为根
        binary_funcs = [f for f in self.function_set if f.arity == 2]
        if not binary_funcs:
            return left_tree + right_tree
        func = random_state.choice(binary_funcs)
        # 前缀表达式：[函数, 左子树..., 右子树...]
        return [func] + left_tree + right_tree

    # def our_subtree_mutation(self, random_state):
    #     """子树变异：随机选择一个内部节点，用一棵维度包含原子树所有维度的新子树替换"""
    #     internal_indices = [i for i, node in enumerate(self.program)
    #                         if callable(node) or isinstance(node, _Function)]
    #     if not internal_indices:
    #         return self.program[:], []
    #
    #     node_idx = random_state.choice(internal_indices)
    #     end_idx, dims = self._parse_subtree(self.program, node_idx)
    #
    #     # 生成一棵至少包含 dims 中所有维度的新子树
    #     new_subtree = self._generate_subtree_containing_dimensions(dims, random_state)
    #     if new_subtree is None:
    #         print("fail subtree!")
    #         return self.program[:], []
    #
    #     # 直接将原子树替换为新子树（不需要再检查维度相等，因为包含关系已满足）
    #     new_program = self.program[:node_idx] + new_subtree + self.program[end_idx:]
    #     return new_program, [node_idx]

    def our_subtree_mutation(self, random_state):
        internal_indices = [i for i, node in enumerate(self.program)
                            if callable(node) or isinstance(node, _Function)]
        if not internal_indices:
            return self.program[:], []

        node_idx = random_state.choice(internal_indices)
        end_idx, dims = self._parse_subtree(self.program, node_idx)

        new_subtree = self._generate_subtree_containing_dimensions(dims, random_state)
        if new_subtree is None:  # 仅当构造失败时（极少数）才打印
            print("subtree generation failed ! skipping mutation.")
            return self.program[:], []

        new_program = self.program[:node_idx] + new_subtree + self.program[end_idx:]
        return new_program, [node_idx]

    # ---------------交叉-----------------------
    def our_crossover(self, other, random_state):
        """维度感知的交叉：找到两个父代中维度集合存在包含关系的子树进行交换"""
        prog1 = self.program[:]
        prog2 = other.program[:]
        # 找出所有内部节点索引
        internal1 = [i for i, node in enumerate(prog1) if callable(node) or isinstance(node, _Function)]
        internal2 = [i for i, node in enumerate(prog2) if callable(node) or isinstance(node, _Function)]
        # 随机打乱顺序
        random_state.shuffle(internal1)
        random_state.shuffle(internal2)

        for idx1 in internal1:
            end1, dims1 = self._parse_subtree(prog1, idx1)
            for idx2 in internal2:
                end2, dims2 = other._parse_subtree(prog2, idx2)
                # 检查包含关系
                if dims1.issubset(dims2) or dims2.issubset(dims1):
                    # 交换子树
                    new_prog1 = prog1[:idx1] + prog2[idx2:end2] + prog1[end1:]
                    new_prog2 = prog2[:idx2] + prog1[idx1:end1] + prog2[end2:]
                    combined_dims = dims1.union(dims2)
                    return new_prog1, new_prog2, (idx1, idx2), combined_dims
        print("fail crossover!")
        return None, None, None, None

    # # -------------hoist mutation--------------
    # def our_hoist_mutation(self, random_state):
    #     """Hoist 变异：随机选择一个内部节点，将该节点提升到根位置（删除外围节点），
    #     然后检查变异后的程序是否仍覆盖所有必需维度。若不满足，则放弃变异，返回原程序。
    #     """
    #     # 复制原程序
    #     new_program = self.program[:]
    #     # 找出所有内部节点（函数）的索引
    #     internal_indices = [i for i, node in enumerate(new_program) if callable(node) or isinstance(node, _Function)]
    #     if not internal_indices:
    #         return new_program, []  # 没有内部节点，无法 hoist
    #
    #     # 随机选择一个内部节点作为提升点
    #     node_idx = random_state.choice(internal_indices)
    #     # 获取该子树的结束索引
    #     end_idx, _ = self._parse_subtree(new_program, node_idx)
    #     # 将该子树提升为整个程序（即截取从 node_idx 到 end_idx 的片段）
    #     hoisted_program = new_program[node_idx:end_idx]
    #
    #     # 检查 hoisted 后的程序是否覆盖所有必需维度
    #     # 创建一个临时 CloneProgram 对象来计算维度（注意：不能直接用 self，因为 self 的 feature_names 等属性是一样的）
    #     # 由于我们需要检查的是 hoisted_program 的维度覆盖，可以直接调用 _check_dimensions 方法，但该方法需要 self 的 required_dimensions 等属性。
    #     # 我们可以创建一个临时实例，或者直接模拟检查。简单起见，我们利用 self._check_dimensions 但传入 hoisted_program 作为程序列表。
    #     # 注意：_check_dimensions 期望 self.feature_names, self.n_features, self.metric_to_dimension, self.required_dimensions 都存在。
    #     # 由于 self 已经拥有这些属性，我们可以直接调用 self._check_dimensions(hoisted_program)。
    #     if self._check_dimensions(hoisted_program):
    #         # 变异成功，返回提升后的程序
    #         return hoisted_program, [node_idx]
    #     else:
    #         print("fail hoist!")
    #         # 维度覆盖不满足，放弃变异，返回原程序
    #         return self.program[:], []
