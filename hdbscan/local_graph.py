import numpy as np
from typing import List, Dict, Tuple
from core.distance import compute_distance_matrix
from core.graph import KruskalMST

class LocalHDBSCANGraph:
    """
    HDBSCAN 局部图构建器。
    负责在 Spark 的 mapPartitions 中计算局部 KNN、核心距离 (Core Distance)、
    互达距离 (MRD)，并构建局部最小生成树 (Local MST)。
    """
    def __init__(self, min_samples: int):
        self.min_samples = min_samples

    def build_local_mst(self, points_data: List[Dict]) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]], Dict[int, float]]:
        """
        构建局部 MST 以及跨界边。
        
        :param points_data: 当前分区内的点列表 (包含 Primary 和 Ghost)
        :return: (local_mst_edges, cross_boundary_edges, core_distances)
                 - local_mst_edges: 局部 MST 边 [(pid1, pid2, weight), ...]
                 - cross_boundary_edges: 跨越分区的边 [(primary_pid, ghost_pid, weight), ...]
                 - core_distances: 核心距离字典 {pid: core_dist}
        """
        if not points_data:
            return [], [], {}
            
        n_points = len(points_data)
        coords = np.array([p['coords'] for p in points_data])
        point_ids = [p['id'] for p in points_data]
        is_ghost = [p['is_ghost'] for p in points_data]
        
        # 1. 计算局部距离矩阵
        dist_matrix = compute_distance_matrix(coords)
        
        # 2. 计算核心距离 (Core Distance)
        # 核心距离是到第 k 个最近邻的距离 (k = min_samples)
        # 注意：包含自身，所以找第 min_samples 个最近的距离
        core_distances = np.zeros(n_points)
        for i in range(n_points):
            # 排序获取距离，如果点数少于 min_samples，核心距离设为无穷大
            if n_points < self.min_samples:
                core_distances[i] = np.inf
            else:
                # np.partition 比 np.sort 更快
                kth_dist = np.partition(dist_matrix[i], self.min_samples - 1)[self.min_samples - 1]
                core_distances[i] = kth_dist
                
        # 3. 计算互达距离矩阵 (MRD)
        # MRD(a, b) = max(core_dist(a), core_dist(b), dist(a, b))
        mrd_matrix = np.maximum(dist_matrix, core_distances[:, np.newaxis])
        mrd_matrix = np.maximum(mrd_matrix, core_distances[np.newaxis, :])
        
        # 4. 构建局部边集合
        local_edges = []
        cross_boundary_edges = []
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                weight = mrd_matrix[i, j]
                
                # 如果两人都是 Ghost，不需要在当前分区建立边，它们的 Primary 所在分区会处理
                if is_ghost[i] and is_ghost[j]:
                    continue
                    
                pid_i = point_ids[i]
                pid_j = point_ids[j]
                
                # 如果一个是 Primary，一个是 Ghost，这是跨界边，单独收集，不参与局部 MST 构建
                if is_ghost[i] != is_ghost[j]:
                    cross_boundary_edges.append((pid_i, pid_j, weight))
                else:
                    # 两人都是 Primary，加入局部图
                    local_edges.append((pid_i, pid_j, weight))
                    
        # 5. 构建局部 MST
        mst_builder = KruskalMST()
        # 局部 Primary 点的数量
        num_primary = sum(1 for g in is_ghost if not g)
        local_mst_edges = mst_builder.build(local_edges, num_nodes=num_primary)
        
        # 转换核心距离为字典返回 (仅返回 Primary 点的，减少网络传输)
        core_dist_dict = {point_ids[i]: core_distances[i] for i in range(n_points) if not is_ghost[i]}
        
        return local_mst_edges, cross_boundary_edges, core_dist_dict
