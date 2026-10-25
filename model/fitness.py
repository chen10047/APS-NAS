# 计算适应度函数

from gplearn.fitness import _Fitness
from joblib import wrap_non_picklable_objects
from scipy import stats
import numpy as np


def make_fitness(*, function, greater_is_better, wrap=True):
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
    if wrap:
        return _Fitness(function=wrap_non_picklable_objects(function),
                        greater_is_better=greater_is_better)
    return _Fitness(function=function,
                    greater_is_better=greater_is_better)


def compute_kendall_many_dataset(y, y_pred, w=None):
    y_0 = np.array([_y.split('+')[0] for _y in y]).astype(float)
    y_1 = np.array([_y.split('+')[1] for _y in y])
    # unique_label = np.unique(y_1)
    indices = np.unique(y_1, return_index=True)[1]
    unique_label = [y_1[i] for i in sorted(indices)]
    corr_scores = []
    for label in unique_label:
        idx = np.argwhere(y_1 == label).reshape(-1)
        _y_pred = y_pred[idx]
        _y = y_0[idx]
        cor = compute_kendall_one_dataset(y=_y, y_pred=_y_pred)
        corr_scores.append(np.round(cor, 6))
    return corr_scores


def compute_kendall_one_dataset(y, y_pred, w=None):
    cor = stats.kendalltau(y, y_pred, nan_policy='omit')[0]
    # print(f"kt: {cor}")
    if np.isnan(cor):
        cor = -1.0
    return cor


def compute_spearman_many_dataset(y, y_pred, w=None):
    y_0 = np.array([_y.split('+')[0] for _y in y]).astype(float)
    y_1 = np.array([_y.split('+')[1] for _y in y])
    # unique_label = np.unique(y_1)
    indices = np.unique(y_1, return_index=True)[1]
    unique_label = [y_1[i] for i in sorted(indices)]
    corr_scores = []
    for label in unique_label:
        idx = np.argwhere(y_1 == label).reshape(-1)
        _y_pred = y_pred[idx]
        _y = y_0[idx]
        cor = compute_spearman_one_dataset(y=_y, y_pred=_y_pred)
        corr_scores.append(np.round(cor, 6))
    return corr_scores


def compute_spearman_one_dataset(y, y_pred, w=None):
    cor = stats.spearmanr(y, y_pred, nan_policy='omit')[0]
    if np.isnan(cor):
        cor = -1.0
    return cor

def compute_topk_s(y, y_pred, k=10, w=None):
    n = len(y)
    k = min(k, n)
    import math
    MIN_TOPK = (2.0 / k) * math.log(math.factorial(k))  # 理论下界
    MAX_TOPK = (2.0 / k) * sum([math.log(n - i) for i in range(k)])

    true_topk_idx = np.argsort(y)[::-1][:k]
    sorted_pred_idx = np.argsort(y_pred)[::-1]
    Lz = [np.where(sorted_pred_idx == idx)[0][0] + 1 for idx in true_topk_idx]

    pred_topk_idx = np.argsort(y_pred)[::-1][:k]
    sorted_true_idx = np.argsort(y)[::-1]
    Lg = [np.where(sorted_true_idx == idx)[0][0] + 1 for idx in pred_topk_idx]

    topk = np.mean(np.log(Lz)) + np.mean(np.log(Lg))

    norm_topk = (topk - MIN_TOPK) / (MAX_TOPK - MIN_TOPK)
    norm_topks = np.clip(norm_topk, 0.0, 1.0)

    return norm_topks


def make_topk_fitness(k=10):
    def topk_func(y, y_pred, w=None):
        return compute_topk_s(y, y_pred, k,w)
    return make_fitness(function=topk_func, greater_is_better=False)


def get_fitness_function(measure='kendall', multiple_dataset=False):
    if multiple_dataset:
        if measure == 'kendall':
            fitness_function = make_fitness(function=compute_kendall_many_dataset, greater_is_better=True)
        elif measure == 'spearman':
            fitness_function = make_fitness(function=compute_spearman_many_dataset, greater_is_better=True)
        else:
            raise NotImplementedError
    else:
        if measure == 'kendall':
            fitness_function = make_fitness(function=compute_kendall_one_dataset, greater_is_better=True)
        elif measure == 'spearman':
            fitness_function = make_fitness(function=compute_spearman_one_dataset, greater_is_better=True)
        elif measure == 'topk_s':
            fitness_function = make_topk_fitness(k=3)
        else:
            raise NotImplementedError
    return fitness_function

