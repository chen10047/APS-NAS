# functions.py
from gplearn.functions import _Function
import numpy as np


def _protected_division(x1, x2):
    """
    保护除法：
    当分母绝对值极小时，返回 1.0（gplearn 经典传统），
    避免产生极其巨大的断崖式惩罚项。
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # 寻找分母接近于 0 的地方
        # 使用 1e-10 规避零或极小值
        res = np.divide(x1, x2)
        # 将 inf 和 nan 替换为 1.0
        res = np.where(np.isinf(res) | np.isnan(res), 1.0, res)
        return res


def _protected_sqrt(x1):
    """
    保护平方根：
    对负数取绝对值后再开方（更适合保留特征趋势），
    或者对负数部分返回 0.0。这里采用更平滑的绝对值法。
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # 做法 A：对负数部分返回 0
        # res = np.sqrt(np.maximum(x1, 0.0))

        # 做法 B：取绝对值开方（推荐，防止信息直接丢失）
        res = np.sqrt(np.abs(x1))
        return res


def _protected_log(x1):
    """
    保护对数：
    必须同时处理 x1 <= 0 的情况（包含 0 产生的 -inf 和负数产生的 nan）
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # 取绝对值，并确保不为 0（最小为 1e-10）
        # 这样既不会产生 -inf，也不会产生 nan
        res = np.log(np.maximum(np.abs(x1), 1e-10))
        return res


# 重新注册函数
our_div = _Function(function=_protected_division, name='div', arity=2)
our_sqrt = _Function(function=_protected_sqrt, name='sqrt', arity=1)
our_log = _Function(function=_protected_log, name='log', arity=1)



# # 在进行数学运算的时候，找到最佳
#
# from gplearn.functions import _Function
# import numpy as np
#
# def _protected_division(x1, x2):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         res = np.divide(x1, x2)
#         res = np.where(np.isinf(res), -10000000, res)
#         res = np.where(np.isnan(res), -10000000, res)  # 新增
#         return res
#
#
# def _protected_sqrt(x1):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         """Closure of square root for negative arguments."""
#         res = np.sqrt(x1)
#         return np.where(np.isnan(res), -10000000, res)
#
#
# def _protected_log(x1):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         res = np.log(x1)
#         """Closure of log for zero and negative arguments."""
#         return np.where(np.isnan(res),  -10000000, res)
#
# our_div = _Function(function=_protected_division, name='div', arity=2)
# our_sqrt = _Function(function=_protected_sqrt, name='sqrt', arity=1)
# our_log = _Function(function=_protected_log, name='log', arity=1)