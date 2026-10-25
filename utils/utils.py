import numpy as np
import json
from scipy import stats
from sklearn.model_selection import train_test_split
#
# # 原有指标列表保留（但实际使用时可能被覆盖）
# METRIC_AGNOSTIC = ['l2_norm', 'params', 'synflow', 'zen', 'val_accuracy']
# METRIC_SPECIFIC = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'nwot', 'plain', 'snip', 'swap', 'meco_opt', 'zico', 'val_accuracy']
# METRICS = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip',
#            'synflow', 'zen', 'swap', 'meco_opt', 'zico', 'val_accuracy']
#
# def evaluate(X, y, gp_model=None, rank_metric='kendall'):
#     # 保持不变
#     for i in range(X.shape[1]):
#         if rank_metric == 'spearman':
#             res = stats.spearmanr(X[:, i], y)
#         else:
#             res = stats.kendalltau(X[:, i], y)
#         print(METRICS[i], round(res[0], 2))
#     our_res = [None, None]
#     if gp_model is not None:
#         new_feature = gp_model.predict(X)
#         if rank_metric == 'spearman':
#             our_res = stats.spearmanr(new_feature, y)
#         else:
#             our_res = stats.kendalltau(new_feature, y)
#         print('new_feature', round(our_res[0], 2))
#     print()
#     return our_res[0]
#
# def modify_y(y, search_space=None, dataset=None, add_suffix=True):
#     """
#     修改 y 值，可选添加后缀以区分数据集。
#     如果 add_suffix=False，直接返回数值数组（转换为列表或保持 ndarray）。
#     """
#     if not add_suffix:
#         # 如果 y 已经是 ndarray，直接返回；否则转换为列表
#         return np.array(y) if isinstance(y, (list, np.ndarray)) else y
#     y_new = []
#     for v in y:
#         y_new.append(f'{v}+{search_space}_{dataset}')
#     return y_new
#
# def create_data(ss, dataset, kind='all', list_metrics=None):
#     """
#     从搜索空间 ss 中读取数据，返回 X, y。
#     如果提供了 list_metrics，则使用该列表；否则使用默认的 METRICS。
#     """
#     list_arch = list(ss[dataset].keys())
#     X = []
#     y = []
#     if list_metrics is None:
#         if kind == 'agnostic':
#             list_metrics = METRIC_AGNOSTIC
#         elif kind == 'specific':
#             list_metrics = METRIC_SPECIFIC
#         else:
#             list_metrics = METRICS
#     for arch in list_arch:
#         x = []
#         for metric in list_metrics:
#             if metric == 'val_accuracy':
#                 y.append(ss[dataset][arch][metric])
#             else:
#                 try:
#                     x.append(ss[dataset][arch][metric]['score'])
#                 except KeyError:
#                     x.append(-9999999)
#         X.append(x)
#     X = np.array(X)
#     y = np.array(y)
#     return X, y
#
# def percentage_split(ss, dataset, kind='all'):
#     # 保持不变
#     X, y = create_data(ss, dataset, kind)
#     data_top = {}
#     length = len(X) // 10
#     idx = np.argsort(-y)
#     X = X[idx]
#     y = y[idx]
#     for i in range(1, 11):
#         if i == 10:
#             _X = X.copy()
#             _y = y.copy()
#         else:
#             _X = X[:length].copy()
#             _y = y[:length].copy()
#         data_top[f'{(i-1)*10}-{i*10}'] = {'X': _X, 'y': _y}
#         X = X[length:].copy()
#         y = y[length:].copy()
#     return data_top
#
# def create_single_data(ss, dataset, seed, train_size=100, kind='all'):
#     X, y = create_data(ss, dataset, kind)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=seed)
#     return X_train, X_test, y_train, y_test
#
# def create_multiple_data(all_ss, list_ss, list_dataset):
#     X, y = [], []
#     for i, _ss in enumerate(list_ss):
#         _X, _y = create_data(all_ss[_ss], list_dataset[i])
#         _y = modify_y(_y, _ss, list_dataset[i], add_suffix=True)
#         if i == 0:
#             X = _X.copy()
#             y = _y.copy()
#         else:
#             X = np.concatenate((X, _X), axis=0)
#             y += _y
#     return X, y
#
# def create_multiple_data_2(all_ss, list_ss, list_dataset, seed, train_size=100, kind='all'):
#     X_train, X_test, y_train, y_test = [], [], [], []
#     for i, _ss in enumerate(list_ss):
#         _X, _y = create_data(all_ss[_ss], list_dataset[i], kind)
#         _y = modify_y(_y, _ss, list_dataset[i], add_suffix=True)
#         _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, train_size=train_size, random_state=seed)
#         if i == 0:
#             X_train = _X_train.copy()
#             X_test = _X_test.copy()
#             y_train = _y_train.copy()
#             y_test = _y_test.copy()
#         else:
#             X_train = np.concatenate((X_train, _X_train), axis=0)
#             X_test = np.concatenate((X_test, _X_test), axis=0)
#             y_train += _y_train
#             y_test += _y_test
#     return X_train, X_test, y_train, y_test
#
# def load_search_spaces(data_path):
#     nb101 = json.load(open(f'{data_path}/NB101/zc_nasbench101.json'))
#     nb201 = json.load(open(f'{data_path}/NB201/zc_nasbench201.json'))
#     nb301 = json.load(open(f'{data_path}/NB301/zc_nasbench301.json'))
#     transnb101_macro = json.load(open(f'{data_path}/TransNAS101/zc_transbench101_macro.json'))
#     transnb101_micro = json.load(open(f'{data_path}/TransNAS101/zc_transbench101_micro.json'))
#
#     search_spaces = {
#         'nb101': nb101,
#         'nb201': nb201,
#         'nb301': nb301,
#         'transnb101_micro': transnb101_micro,
#         'transnb101_macro': transnb101_macro
#     }
#     return search_spaces


#--------------------------------------------
import numpy as np
import json
from scipy import stats
from sklearn.model_selection import train_test_split

# 原有指标列表保留（但实际使用时可能被覆盖）
METRIC_AGNOSTIC = ['l2_norm', 'params', 'synflow', 'zen', 'val_accuracy']
METRIC_SPECIFIC = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'nwot', 'plain', 'snip', 'swap', 'meco_opt', 'zico', 'val_accuracy']
METRICS = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip',
           'synflow', 'zen', 'swap', 'meco_opt', 'zico', 'val_accuracy']

def evaluate(X, y, gp_model=None, rank_metric='kendall'):
    # 保持不变
    for i in range(X.shape[1]):
        if rank_metric == 'spearman':
            res = stats.spearmanr(X[:, i], y)
        else:
            res = stats.kendalltau(X[:, i], y)
        print(METRICS[i], round(res[0], 2))
    our_res = [None, None]
    if gp_model is not None:
        new_feature = gp_model.predict(X)
        if rank_metric == 'spearman':
            our_res = stats.spearmanr(new_feature, y)
        else:
            our_res = stats.kendalltau(new_feature, y)
        print('new_feature', round(our_res[0], 2))
    print()
    return our_res[0]

def modify_y(y, search_space=None, dataset=None, add_suffix=True):
    """
    修改 y 值，可选添加后缀以区分数据集。
    如果 add_suffix=False，直接返回数值数组（转换为列表或保持 ndarray）。
    """
    if not add_suffix:
        # 如果 y 已经是 ndarray，直接返回；否则转换为列表
        return np.array(y) if isinstance(y, (list, np.ndarray)) else y
    y_new = []
    for v in y:
        y_new.append(f'{v}+{search_space}_{dataset}')
    return y_new

def create_data(ss, dataset, kind='all', list_metrics=None):
    """
    从搜索空间 ss 中读取数据，返回 X, y。
    如果提供了 list_metrics，则使用该列表；否则使用默认的 METRICS。
    """
    target_metric = 'val_accuracy'
    list_arch = list(ss[dataset].keys())
    X = []
    y = []
    if list_metrics is None:
        if kind == 'agnostic':
            list_metrics = METRIC_AGNOSTIC
        elif kind == 'specific':
            list_metrics = METRIC_SPECIFIC
        else:
            list_metrics = METRICS
    for arch in list_arch:
        x = []
        for metric in list_metrics:
            if metric == 'val_accuracy':
                continue
                # y.append(ss[dataset][arch][metric])
            else:
                try:
                    x.append(ss[dataset][arch][metric]['score'])
                except KeyError:
                    x.append(-9999999)
        X.append(x)
        y.append(ss[dataset][arch][target_metric])
    X = np.array(X)
    y = np.array(y)
    return X, y

def percentage_split(ss, dataset, kind='all'):
    X, y = create_data(ss, dataset, kind)
    data_top = {}
    length = len(X) // 10
    idx = np.argsort(-y)
    X = X[idx]
    y = y[idx]
    for i in range(1, 11):
        if i == 10:
            _X = X.copy()
            _y = y.copy()
        else:
            _X = X[:length].copy()
            _y = y[:length].copy()
        data_top[f'{(i-1)*10}-{i*10}'] = {'X': _X, 'y': _y}
        X = X[length:].copy()
        y = y[length:].copy()
    return data_top

def create_single_data(ss, dataset, seed, train_size=100, kind='all'):
    X, y = create_data(ss, dataset, kind)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=seed)
    return X_train, X_test, y_train, y_test

def create_multiple_data(all_ss, list_ss, list_dataset):
    X, y = [], []
    for i, _ss in enumerate(list_ss):
        _X, _y = create_data(all_ss[_ss], list_dataset[i])
        _y = modify_y(_y, _ss, list_dataset[i], add_suffix=True)
        if i == 0:
            X = _X.copy()
            y = _y.copy()
        else:
            X = np.concatenate((X, _X), axis=0)
            y += _y
    return X, y

def create_multiple_data_2(all_ss, list_ss, list_dataset, seed, train_size=100, kind='all'):
    X_train, X_test, y_train, y_test = [], [], [], []
    for i, _ss in enumerate(list_ss):
        _X, _y = create_data(all_ss[_ss], list_dataset[i], kind)
        _y = modify_y(_y, _ss, list_dataset[i], add_suffix=True)
        _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, train_size=train_size, random_state=seed)
        if i == 0:
            X_train = _X_train.copy()
            X_test = _X_test.copy()
            y_train = _y_train.copy()
            y_test = _y_test.copy()
        else:
            X_train = np.concatenate((X_train, _X_train), axis=0)
            X_test = np.concatenate((X_test, _X_test), axis=0)
            y_train += _y_train
            y_test += _y_test
    return X_train, X_test, y_train, y_test

def load_search_spaces(data_path):
    nb101 = json.load(open(f'{data_path}/NB101/zc_nasbench101.json'))
    nb201 = json.load(open(f'{data_path}/NB201/zc_nasbench201.json'))
    nb301 = json.load(open(f'{data_path}/NB301/zc_nasbench301.json'))
    transnb101_macro = json.load(open(f'{data_path}/TransNAS101/zc_transbench101_macro.json'))
    transnb101_micro = json.load(open(f'{data_path}/TransNAS101/zc_transbench101_micro.json'))
    darts = json.load(open(f'{data_path}/darts/zc_darts.json'))

    search_spaces = {
        'nb101': nb101,
        'nb201': nb201,
        'nb301': nb301,
        'transnb101_micro': transnb101_micro,
        'transnb101_macro': transnb101_macro,
        'darts':darts
    }
    return search_spaces

import numpy as np
import random
import torch

from .compare import find_the_better
# from nasbench import wrap_api as api
# wrap_api = api.NASBench_()
try:
    from nasbench import wrap_api as api
    HAS_NASBENCH = True
except ImportError:
    HAS_NASBENCH = False
    api = None

def check_valid(hash_key, **kwargs):
    """
    - Check if the current solution already exists on the set of checklists.
    """
    return np.all([hash_key not in kwargs[L] for L in kwargs])

def X2matrices(X):
    """
    - This function in used to convert the vector which used to representation the architecture to 2 matrix (Edges matrix & Operators matrix).
    - This function is used to help for getting hash key in '101' problems.
    """
    if not HAS_NASBENCH:
        raise RuntimeError("nasbench not installed, cannot call X2matrices")
    IDX_OPS = [1, 3, 6, 10, 15]
    edges_matrix = np.zeros((7, 7), dtype=np.int8)
    for row in range(6):
        idx_list = None
        if row == 0:
            idx_list = [2, 4, 7, 11, 16, 22]
        elif row == 1:
            idx_list = [5, 8, 12, 17, 23]
        elif row == 2:
            idx_list = [9, 13, 18, 24]
        elif row == 3:
            idx_list = [14, 19, 25]
        elif row == 4:
            idx_list = [20, 26]
        elif row == 5:
            idx_list = [27]
        for i, edge in enumerate(idx_list):
            if X[edge] - 1 == 0:
                edges_matrix[row][row + i + 1] = 1

    ops_matrix = ['input']
    for i in IDX_OPS:
        if X[i] == 2:
            ops_matrix.append('conv1x1-bn-relu')
        elif X[i] == 3:
            ops_matrix.append('conv3x3-bn-relu')
        else:
            ops_matrix.append('maxpool3x3')
    ops_matrix.append('output')

    return edges_matrix, ops_matrix

def insert_to_list(X):
    indices = [4, 8, 12]

    for idx, i in enumerate(indices):
        X.insert(i + idx, '|')
        idx += 1
    return X

def remove_values_from_list(x, val):
    return [value for value in x if value != val]

def get_hashKey(arch, problem_name):
    """
    This function is used to get the hash key of architecture. The hash key is used to avoid the existence of duplication in the population.\n
    - *Output*: The hash key of architecture.
    """
    if problem_name in ['NASBench101', 'NASBench201']:
        if not HAS_NASBENCH:
            raise RuntimeError(f"nasbench not installed, cannot get hashKey for {problem_name}")
        arch_dummy = arch.copy()
        if problem_name == 'NASBench101':
            edges_matrix, ops_matrix = X2matrices(arch_dummy)
            model_spec = api.ModelSpec(edges_matrix, ops_matrix)
            hashKey = wrap_api.get_module_hash(model_spec)
        elif problem_name == 'NASBench201':
            hashKey = ''.join(map(str, arch_dummy))
        else:
            raise ValueError(f'Not supporting this problem - {problem_name}.')
        return hashKey
    else:
        # 对于非 NAS 问题，返回一个基于字符串的简单哈希
        return str(hash(tuple(arch))) if isinstance(arch, (list, tuple)) else str(hash(arch))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

def get_front_0(F):
    l = len(F)
    r = np.zeros(l, dtype=np.int8)
    for i in range(l):
        if r[i] == 0:
            for j in range(i + 1, l):
                better_sol = find_the_better(F[i], F[j])
                if better_sol == 0:
                    r[j] += 1
                elif better_sol == 1:
                    r[i] += 1
                    break
    return r == 0