# run_nsga2_tnb101_micro.py
import os, sys, json, argparse, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.problem import nsga2Problem
from algorithms.nsga2 import NSGAII
from model.RankAndCrowdingSurvival import RankAndCrowdingSurvival
from utils.utils import load_search_spaces, create_data, modify_y
from sklearn.model_selection import train_test_split

# 导入评估函数
from model.fitness import compute_kendall_one_dataset, compute_topk_s

# ========== 参数配置 ==========
TRAIN_SIZE = 600
DATA_PATH = 'database'
METRICS = [
    'nwot',  'zen',
    'zico', 'meco_opt', 'snip', 'sweetinator',
    'ni', 'ntk_mse',
    'flops', 'params',
    'tnb101_micro_identity_ratio', 'tnb101_micro_zero_ratio', 'tnb101_micro_conv3x3_ratio','tnb101_micro_conv1x1_ratio'
]

METRIC_TO_DIMENSION = {
    # 'aze': 'expressivity',
    'nwot': 'expressivity',
    'zen': 'expressivity',
    # 'singular': 'expressivity',
    'snip': 'trainability',
    # 'ntk': 'trainability',
    'meco_opt': 'trainability',
    'zico': 'trainability',
    'sweetinator': 'trainability',
    'ni': 'generalization',
    'ntk_mse': 'generalization',
    'flops': 'complexity',
    'params': 'complexity',
    'tnb101_micro_identity_ratio': 'spatial',
    'tnb101_micro_zero_ratio': 'spatial',
    'tnb101_micro_conv3x3_ratio': 'spatial',
    'tnb101_micro_conv1x1_ratio':'spatial',
}
REQUIRED_DIMENSIONS = {'expressivity', 'trainability', 'generalization', 'complexity', 'spatial'}
# REQUIRED_DIMENSIONS = {'expressivity', 'trainability', 'generalization', 'complexity'}

POP_SIZE = 100
N_GEN = 50
P_CROSSOVER = 0.7
P_SUBTREE_MUTATION = 0.2
P_POINT_MUTATION = 0.1
FUNCTION_SET = ['add', 'sub', 'mul', 'neg', 'div', 'log', 'sqrt']
INIT_DEPTH = (3, 5)
TOPK = 3
def compute_fused(kt, topks, lambda_val):
    """计算 fused = kt - lambda * topks"""
    return kt - lambda_val * topks

def evaluate_on_dataset(individual, X, y, k=TOPK):
    y_pred = individual.X.execute(X)
    # print(f"len:{len(y_pred)}")
    kt = compute_kendall_one_dataset(y, y_pred)
    topks = compute_topk_s(y, y_pred, k=k)
    return kt, topks, y_pred

def get_knee_point(pareto_individuals, fused_values, values):
    n = len(pareto_individuals)
    if n == 0:
        return None
    if n == 1:
        return pareto_individuals[0]

    fused_arr = np.array(fused_values)
    leaf_arr = np.array(values)

    min_f, max_f = fused_arr.min(), fused_arr.max()
    min_l, max_l = leaf_arr.min(), leaf_arr.max()

    fused_norm = (fused_arr - min_f) / (max_f - min_f) if max_f - min_f > 1e-10 else np.zeros(n)
    leaf_norm = (max_l - leaf_arr) / (max_l - min_l) if max_l - min_l > 1e-10 else np.zeros(n)

    distances = (1.0 - fused_norm) + (1.0 - leaf_norm)

    min_dist = np.min(distances)
    candidate_indices = np.where(np.abs(distances - min_dist) < 1e-9)[0]

    if len(candidate_indices) == 1:
        return pareto_individuals[candidate_indices[0]]

    best_idx = candidate_indices[0]
    best_fused = fused_values[best_idx]
    best_leaf = values[best_idx]

    for idx in candidate_indices[1:]:
        current_fused = fused_values[idx]
        current_leaf = values[idx]
        # Fused 更大更好；若 Fused 相同，Leaf 更小更好
        if (current_fused > best_fused) or (abs(current_fused - best_fused) < 1e-9 and current_leaf < best_leaf):
            best_idx = idx
            best_fused = current_fused
            best_leaf = current_leaf

    return pareto_individuals[best_idx]


def run_single_experiment(search_space, dataset, seed, output_dir, lambda_val=1.0):
    print(f"\n===== Run: {search_space}/{dataset}, seed={seed} =====")

    # 1. 加载数据
    all_ss = load_search_spaces(DATA_PATH)
    ss = all_ss[search_space]
    X_full, y_full = create_data(ss, dataset, list_metrics=METRICS)
    y_full = modify_y(y_full, search_space, dataset, add_suffix=False)

    # 2. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, train_size=TRAIN_SIZE, random_state=seed
    )
    print(f"Training arch count: {len(X_train)}")
    print(f"Test arch count: {len(X_test)}")
    print(f"Full arch count: {len(X_full)}")

    # 3. 定义问题
    problem = nsga2Problem(
        X_train=X_train,
        y_train=y_train,
        topk=TOPK,
        Lambda=lambda_val
    )
    problem.feature_names = METRICS

    # 4. 配置算法
    algorithm = NSGAII(
        name="NSGAII",
        pop_size=POP_SIZE,
        n_generations=N_GEN,
        function_set=FUNCTION_SET,
        init_depth=INIT_DEPTH,
        const_range=None,
        p_crossover=P_CROSSOVER,
        p_subtree_mutation=P_SUBTREE_MUTATION,
        p_point_mutation=P_POINT_MUTATION,
        metric_to_dimension=METRIC_TO_DIMENSION,
        required_dimensions=REQUIRED_DIMENSIONS,
        debug=True,
        max_depth=6,
        early_stop_patience=0,
        early_stop_threshold=0.0005
    )
    survival = RankAndCrowdingSurvival()
    algorithm.set_hyperparameters(survival=survival)

    # 5. 运行优化
    start_time = time.time()
    algorithm.solve(problem, seed=seed)
    end_time = time.time()
    elapsed_seconds = end_time - start_time

    # 6. 提取结果
    final_pop = algorithm.pop

    # Pareto 前沿 (rank=0)
    front0_indices = [i for i, ind in enumerate(final_pop) if ind.get('rank') == 0]
    pareto_front = [final_pop[i] for i in front0_indices]

    # 按表达式去重，保留第一次出现的个体
    seen_exprs = set()
    deduped_front = []
    for ind in pareto_front:
        expr = str(ind.X)
        if expr not in seen_exprs:
            seen_exprs.add(expr)
            deduped_front.append(ind)
    pareto_front = deduped_front

    def extract_info(ind, on_train=True):
        prog = ind.X
        if on_train:
            neg_fused, leaf_count_val = ind.F
            fused = -neg_fused
        else:
            fused = None
            leaf_count_val = None
        return {
            'fused': float(fused) if fused is not None else None,
            'leaf_count': float(leaf_count_val) if leaf_count_val is not None else None,
            'depth': prog.depth_,
            'length': prog.length_,
            'expression': str(prog),
        }

    # ---- 从 Pareto 前沿中找出 3 个特殊个体 ----
    pf_train_info = []
    for ind in pareto_front:
        kt_train, topks_train, _ = evaluate_on_dataset(ind, X_train, y_train)
        leaf_count_val = ind.F[1]
        fused_val = compute_fused(kt_train, topks_train, lambda_val)
        pf_train_info.append({
            'ind': ind,
            'kt': kt_train,
            'topks': topks_train,
            'fused': fused_val,
            'leaf_count': leaf_count_val,
        })

    if len(pf_train_info) > 0:
        max_fused_entry = max(pf_train_info, key=lambda x: x['fused'])
        best_fused_ind = max_fused_entry['ind']

        min_leaf_entry = min(pf_train_info, key=lambda x: x['leaf_count'])
        best_leaf_ind = min_leaf_entry['ind']

        fused_vals = [info['fused'] for info in pf_train_info]
        leaf_count_vals = [info['leaf_count'] for info in pf_train_info]
        knee_ind = get_knee_point(pareto_front, fused_vals, leaf_count_vals)
    else:
        best_fused_ind = None
        best_leaf_ind = None
        knee_ind = None

    def evaluate_individual_on_test(ind, X_test, y_test, lambda_val):
        """评估个体在测试集上的 KT, topks, top1_acc"""
        if ind is None:
            return None
        kt_test, topks_test, y_pred_test = evaluate_on_dataset(ind, X_test, y_test)
        idx_max_pred_test = np.argmax(y_pred_test)
        true_acc_max_test = y_test[idx_max_pred_test]
        return {
            'kendall_tau_test': float(kt_test),
            'topks_test': float(topks_test),
            'top1_accuracy': float(true_acc_max_test)
        }

    best_fused_test = evaluate_individual_on_test(best_fused_ind, X_test, y_test, lambda_val)
    best_leaf_test = evaluate_individual_on_test(best_leaf_ind, X_test, y_test, lambda_val)
    knee_test = evaluate_individual_on_test(knee_ind, X_test, y_test, lambda_val)

    def get_train_info(entry):
        if entry is None:
            return None
        return {
            'fused': float(entry['fused']),
            'kendall_tau': float(entry['kt']),
            'topks': float(entry['topks']),
            'leaf_count': float(entry['leaf_count']),
            'length': entry['ind'].X.length_,
            'depth': entry['ind'].X.depth_,
            'expression': str(entry['ind'].X),
        }

    best_fused_train = get_train_info(max_fused_entry) if len(pf_train_info) > 0 else None
    best_leaf_train = get_train_info(min_leaf_entry) if len(pf_train_info) > 0 else None

    knee_train_entry = None
    if knee_ind is not None and len(pf_train_info) > 0:
        for info in pf_train_info:
            if info['ind'] is knee_ind:
                knee_train_entry = get_train_info(info)
                break

    pareto_front_with_test = []
    for ind in pareto_front:
        base_info = extract_info(ind, on_train=True)
        test_info = evaluate_individual_on_test(ind, X_test, y_test, lambda_val)
        if test_info is not None:
            full_info = {
                **base_info,
                'test_kt': test_info['kendall_tau_test'],
                'test_topks': test_info['topks_test'],
                'test_top1': test_info['top1_accuracy'],
            }
        else:
            full_info = {
                **base_info,
                'test_kt': None,
                'test_topks': None,
                'test_top1': None,
            }
        pareto_front_with_test.append(full_info)

    results = {
        'seed': seed,
        'lambda': lambda_val,
        'arch_number': len(X_train),
        'runtime(s)': elapsed_seconds,
        'topk': TOPK,
        'pareto_front_size': len(pareto_front),
        'pareto_front': pareto_front_with_test,
        'best_fused_train': best_fused_train,
        'best_fused_test': best_fused_test,
        'best_leaf_count_train': best_leaf_train,
        'best_leaf_count_test': best_leaf_test,
        'knee_point_train': knee_train_entry,
        'knee_point_test': knee_test,
    }

    # 7. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{search_space}_{dataset}_seed{seed}.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_space', type=str, required=True,
                        help='e.g., nb101, nb201, nb301, transnb101_micro, transnb101_macro')
    parser.add_argument('--dataset', type=str, required=True,
                        help='e.g., cifar10, cifar100, imagenet16-120, class_scene')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lambda', type=float, dest='lambda_val', default=1.0, help='lambda parameter for fused = kt - lambda*topks')
    parser.add_argument('--output_dir', type=str, default='exp_sr_nsga2', help='output directory')
    args = parser.parse_args()

    run_single_experiment(args.search_space, args.dataset, args.seed, args.output_dir, args.lambda_val)

if __name__ == '__main__':
    main()
