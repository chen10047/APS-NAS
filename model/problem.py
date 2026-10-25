# gp_problem.py
import numpy as np
from model.fitness import compute_kendall_one_dataset, compute_topk_s

class nsga2Problem:
    """
    NSGA-II
    目标1: 最大化 fused = kt - Lambda * topks (转化为最小化负值)
    目标2: 最小化 complexity = depth/max_depth + leaf_count/max_leaf
    # 旧目标(leaf): 目标2: 最小化 叶子结点个数 leaf_count
    # 旧目标(depth): 目标2: 最小化 depth
    """
    def __init__(self, X_train, y_train, topk=10, Lambda=1.0, max_depth=8):
        """
        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
            训练数据特征（各种零成本指标 + 空间特征）
        y_train : np.ndarray, shape (n_samples,)
            真实性能（例如验证准确率）
        max_eval : int
            最大评估次数（终止条件）
            topk : int
                Topks 公式中的阈值
        Lambda : float
            fused = kt - Lambda * topks 中的 Lambda 权重
        max_depth : int
            树的最大深度，max_leaf = 2 ** max_depth
        """
        self.X_train = X_train
        self.y_train = y_train
        # self.max_eval = max_eval
        self.topk = topk
        self.Lambda = Lambda
        self.max_depth = max_depth
        self.max_leaf = 2 ** max_depth
        self.n_eval = 0   # 当前已评估的个体数

        self.name = "nsga2"
        self.dataset = "cifar10"
        self.objective_0 = "neg_fused"   # 第一个目标（最小化）
        # self.objective_1 = "complexity"       # 第二个目标（最小化）: depth/max_depth + leaf/max_leaf
        # 旧目标(leaf):
        self.objective_1 = "leaf_count"
        # 旧目标(depth):
        # self.objective_1 = "depth"

    def evaluate(self, arch, **kwargs):
        """
        评估一个表达式树个体。

        Parameters
        ----------
        arch : CloneProgram 实例
            待评估的表达式树
        **kwargs : 忽略额外参数（为兼容 NSGA-II 框架的调用方式）

        Returns
        -------
        list of float
            [ -(kt - Lambda * topks), depth_/max_depth + leaf_count_/max_leaf ]  两个目标值，均需最小化
        """
        # 预测所有训练样本的值
        y_pred = arch.execute(self.X_train)
        # print(f"len(y_pred) = {len(y_pred)}")

        kt = compute_kendall_one_dataset(self.y_train, y_pred)
        topks = compute_topk_s(self.y_train, y_pred, k=self.topk)

        self.n_eval += 1

        # return [-kt, arch.leaf_count_]
        # return [topks, arch.leaf_count_]
        # complexity = arch.depth_ / self.max_depth + 1.8 * arch.leaf_count_ / self.max_leaf
        # return [-(kt - self.Lambda * topks), complexity]

        # leaf:
        return [-(kt - self.Lambda * topks), arch.leaf_count_]
        # depth:
        # return [-(kt - self.Lambda * topks), arch.depth_]


    def reset(self):
        self.n_eval = 0
