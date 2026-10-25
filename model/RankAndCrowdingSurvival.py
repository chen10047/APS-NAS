
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort

class RankAndCrowdingSurvival:
    def __init__(self):
        self.name = 'Rank and Crowding Survival'

    @staticmethod
    def do(pop, n_survive):
        # 兼容列表和 Population 对象
        if hasattr(pop, 'get'):
            F = pop.get('F')
        else:
            F = np.array([ind.F for ind in pop])

        # 非支配排序
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)
        print(f"F shape: {F.shape}, F min/max: {F.min(axis=0)} / {F.max(axis=0)}")
        print(f"fronts: {fronts}")

        survivors = []
        for k, front in enumerate(fronts):
            # 计算拥挤距离
            crowding_of_front = calculating_crowding_distance(F[front, :])
            # 保存 rank 和 crowding
            for j, i in enumerate(front):
                pop[i].set('rank', k)
                pop[i].set('crowding', crowding_of_front[j])

            if len(survivors) + len(front) > n_survive:
                # I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = np.argsort(-crowding_of_front, kind='stable')
                I = I[:(n_survive - len(survivors))]
            else:
                I = np.arange(len(front))

            survivors.extend(front[I])

        # 返回与原 pop 相同类型的结果
        if isinstance(pop, list):
            return [pop[i] for i in survivors]
        else:
            return pop[survivors]

def calculating_crowding_distance(F):
    infinity = 1e+14
    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:
        I = np.argsort(F, axis=0, kind='mergesort')
        F_sorted = F[I, np.arange(n_obj)]
        dist = np.concatenate([F_sorted, np.full((1, n_obj), np.inf)]) - np.concatenate([np.full((1, n_obj), -np.inf), F_sorted])
        index_dist_is_zero = np.where(dist == 0)
        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]
        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]
        norm = np.max(F_sorted, axis=0) - np.min(F_sorted, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0
        J = np.argsort(I, axis=0)
        crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj
    crowding[np.isinf(crowding)] = infinity
    return crowding

# #---------------原版----------
# # import numpy as np
# #
# # # from pymoo.util.non_dominated_sorting import NonDominatedSorting
# # from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
# # from pymoo.util.randomized_argsort import randomized_argsort
# #
# #
# # class RankAndCrowdingSurvival:
# #     def __init__(self):
# #         self.name = 'Rank and Crowding Survival'
# #
# #     @staticmethod
# #     def do(pop, n_survive):
# #         # get the objective space values and objects
# #         F = pop.get('F')
# #
# #         # the final indices of surviving individuals
# #         survivors = []
# #
# #         # do the non-dominated sorting until splitting front
# #         # fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)
# #         nds = NonDominatedSorting()
# #         fronts = nds(F, n_stop_if_ranked=n_survive)
# #         for k, front in enumerate(fronts):
# #
# #             # calculate the crowding distance of the front
# #             crowding_of_front = calculating_crowding_distance(F[front, :])
# #
# #             # save rank and crowding in the individual class
# #             for j, i in enumerate(front):
# #                 pop[i].set('rank', k)
# #                 pop[i].set('crowding', crowding_of_front[j])
# #
# #             # current front sorted by crowding distance if splitting
# #             if len(survivors) + len(front) > n_survive:
# #                 I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
# #                 I = I[:(n_survive - len(survivors))]
# #
# #             # otherwise take the whole front unsorted
# #             else:
# #                 I = np.arange(len(front))
# #
# #             # extend the survivors by all or selected individuals
# #             survivors.extend(front[I])
# #         return pop[survivors]
# #
# #
# # def calculating_crowding_distance(F):
# #     infinity = 1e+14
# #
# #     n_points = F.shape[0]
# #     n_obj = F.shape[1]
# #
# #     if n_points <= 2:
# #         return np.full(n_points, infinity)
# #     else:
# #
# #         # sort each column and get index
# #         I = np.argsort(F, axis=0, kind='mergesort')
# #
# #         # now really sort the whole array
# #         F = F[I, np.arange(n_obj)]
# #
# #         # get the distance to the last element in sorted list and replace zeros with actual values
# #         dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) - np.concatenate([np.full((1, n_obj), -np.inf), F])
# #
# #         index_dist_is_zero = np.where(dist == 0)
# #
# #         dist_to_last = np.copy(dist)
# #         for i, j in zip(*index_dist_is_zero):
# #             dist_to_last[i, j] = dist_to_last[i - 1, j]
# #
# #         dist_to_next = np.copy(dist)
# #         for i, j in reversed(list(zip(*index_dist_is_zero))):
# #             dist_to_next[i, j] = dist_to_next[i + 1, j]
# #
# #         # normalize all the distances
# #         norm = np.max(F, axis=0) - np.min(F, axis=0)
# #         norm[norm == 0] = np.nan
# #         dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm
# #
# #         # if we divided by zero because all values in one columns are equal replace by none
# #         dist_to_last[np.isnan(dist_to_last)] = 0.0
# #         dist_to_next[np.isnan(dist_to_next)] = 0.0
# #
# #         # sum up the distance to next and last and norm by objectives - also reorder from sorted list
# #         J = np.argsort(I, axis=0)
# #         crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj
# #
# #     # replace infinity with a large number
# #     crowding[np.isinf(crowding)] = infinity
# #     return crowding
#
#
# # import numpy as np
# #
# #
# # class RankAndCrowdingSurvival:
# #     def __init__(self):
# #         self.name = 'Rank and Crowding Survival'
# #
# #     @staticmethod
# #     def do(pop, n_survive):
# #         # 提取目标值 F
# #         if hasattr(pop, 'get'):
# #             F = pop.get('F')
# #         else:
# #             F = np.array([ind.F for ind in pop])
# #
# #         n = len(F)
# #         if n <= n_survive:
# #             return pop
# #
# #         fronts = non_dominated_sort(F)
# #         print(f"fronts: {[len(f) for f in fronts]}")
# #         survivors = []
# #         for front in fronts:
# #             front = np.array(front)  # 添加这一行
# #             if len(survivors) + len(front) <= n_survive:
# #                 survivors.extend(front)
# #             else:
# #                 crowding = crowding_distance(F[front])
# #                 idx_sorted = np.argsort(crowding)[::-1]
# #                 needed = n_survive - len(survivors)
# #                 survivors.extend(front[idx_sorted[:needed]])
# #                 break
# #
# #         if isinstance(pop, list):
# #             return [pop[i] for i in survivors]
# #         else:
# #             return pop[survivors]
# #
# #
# # def non_dominated_sort(F):
# #     n = len(F)
# #     domination_count = np.zeros(n, dtype=int)
# #     dominated_by = [[] for _ in range(n)]
# #     fronts = []
# #     front0 = []
# #     for i in range(n):
# #         for j in range(n):
# #             if i == j:
# #                 continue
# #             if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
# #                 dominated_by[i].append(j)
# #             elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
# #                 domination_count[i] += 1
# #         if domination_count[i] == 0:
# #             front0.append(i)
# #     fronts.append(front0)
# #     current_front = front0
# #     while current_front:
# #         next_front = []
# #         for i in current_front:
# #             for j in dominated_by[i]:
# #                 domination_count[j] -= 1
# #                 if domination_count[j] == 0:
# #                     next_front.append(j)
# #         fronts.append(next_front)
# #         current_front = next_front
# #     if fronts and not fronts[-1]:
# #         fronts.pop()
# #     return fronts
# #
# #
# # def crowding_distance(F):
# #     n_points, n_obj = F.shape
# #     if n_points <= 2:
# #         return np.full(n_points, np.inf)
# #     distances = np.zeros(n_points)
# #     for obj in range(n_obj):
# #         idx = np.argsort(F[:, obj])
# #         distances[idx[0]] = np.inf
# #         distances[idx[-1]] = np.inf
# #         f_min = F[idx[0], obj]
# #         f_max = F[idx[-1], obj]
# #         if f_max == f_min:
# #             continue
# #         for i in range(1, n_points - 1):
# #             distances[idx[i]] += (F[idx[i + 1], obj] - F[idx[i - 1], obj]) / (f_max - f_min)
# #     return distances
#
# #-----------------
# # import numpy as np
# #
# # class RankAndCrowdingSurvival:
# #     def __init__(self):
# #         self.name = 'Rank and Crowding Survival'
# #
# #     @staticmethod
# #     def do(pop, n_survive):
# #         # 提取目标值 F
# #         if hasattr(pop, 'get'):
# #             F = pop.get('F')
# #         else:
# #             F = np.array([ind.F for ind in pop])
# #
# #         n = len(F)
# #         if n <= n_survive:
# #             # 为所有个体设置 rank=0
# #             for ind in pop:
# #                 ind.set('rank', 0)
# #             return pop
# #
# #         fronts = non_dominated_sort(F)
# #         print(f"F shape: {F.shape}, F min/max: {F.min(axis=0)} / {F.max(axis=0)}")
# #         print(f"fronts: {fronts}")
# #         survivors = []
# #         for k, front in enumerate(fronts):
# #             front_arr = np.array(front)
# #             # 设置 rank
# #             for idx in front_arr:
# #                 pop[idx].set('rank', k)
# #             # 计算拥挤距离
# #             if len(front_arr) > 0:
# #                 crowd = crowding_distance(F[front_arr])
# #                 for j, idx in enumerate(front_arr):
# #                     pop[idx].set('crowding', crowd[j])
# #             # 选择幸存者
# #             if len(survivors) + len(front_arr) <= n_survive:
# #                 survivors.extend(front_arr)
# #             else:
# #                 if len(front_arr) > 1:
# #                     crowd = crowding_distance(F[front_arr])
# #                     sorted_idx = np.argsort(crowd)[::-1]
# #                     needed = n_survive - len(survivors)
# #                     survivors.extend(front_arr[sorted_idx[:needed]])
# #                 else:
# #                     survivors.extend(front_arr[:n_survive - len(survivors)])
# #                 break
# #
# #         if isinstance(pop, list):
# #             return [pop[i] for i in survivors]
# #         else:
# #             return pop[survivors]
# #
# # def non_dominated_sort(F):
# #     n = len(F)
# #     domination_count = np.zeros(n, dtype=int)
# #     dominated_by = [[] for _ in range(n)]
# #     fronts = []
# #     front0 = []
# #     for i in range(n):
# #         for j in range(n):
# #             if i == j:
# #                 continue
# #             if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
# #                 dominated_by[i].append(j)
# #             elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
# #                 domination_count[i] += 1
# #         if domination_count[i] == 0:
# #             front0.append(i)
# #     fronts.append(front0)
# #     current_front = front0
# #     while current_front:
# #         next_front = []
# #         for i in current_front:
# #             for j in dominated_by[i]:
# #                 domination_count[j] -= 1
# #                 if domination_count[j] == 0:
# #                     next_front.append(j)
# #         fronts.append(next_front)
# #         current_front = next_front
# #     if fronts and not fronts[-1]:
# #         fronts.pop()
# #     return fronts
# #
# # def crowding_distance(F):
# #     n_points, n_obj = F.shape
# #     if n_points <= 2:
# #         return np.full(n_points, np.inf)
# #     distances = np.zeros(n_points)
# #     for obj in range(n_obj):
# #         idx = np.argsort(F[:, obj])
# #         distances[idx[0]] = np.inf
# #         distances[idx[-1]] = np.inf
# #         f_min = F[idx[0], obj]
# #         f_max = F[idx[-1], obj]
# #         if f_max == f_min:
# #             continue
# #         for i in range(1, n_points - 1):
# #             distances[idx[i]] += (F[idx[i+1], obj] - F[idx[i-1], obj]) / (f_max - f_min)
# #     return distances