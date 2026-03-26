import numpy as np
from typing import List, Dict, Tuple, Set
from core.distance import compute_distance_matrix

class LocalDBSCAN:
    """
    单机版局部 DBSCAN 引擎，设计用于在 Spark 的 mapPartitions 中运行。
    """
    def __init__(self, eps: float, min_samples: int):
        self.eps = eps
        self.min_samples = min_samples
        
    def fit(self, points_data: List[Dict]) -> Tuple[Dict[int, int], List[int]]:
        """
        在局部数据上运行 DBSCAN。
        
        :param points_data: 包含点信息的字典列表，例如 [{'id': 1, 'coords': array([x, y]), 'is_ghost': False}, ...]
        :return: (labels_dict, core_points_list)
                 labels_dict: {point_id: local_cluster_id} (噪声点 cluster_id = -1)
                 core_points_list: 核心点的 point_id 列表
        """
        if not points_data:
            return {}, []
            
        n_points = len(points_data)
        coords = np.array([p['coords'] for p in points_data])
        point_ids = [p['id'] for p in points_data]
        is_ghost = [p['is_ghost'] for p in points_data]
        
        # 计算距离矩阵 (局部向量化计算)
        dist_matrix = compute_distance_matrix(coords)
        
        # 寻找邻居
        neighbors = []
        for i in range(n_points):
            # 找到距离 <= eps 的点的索引
            neighbor_indices = np.where(dist_matrix[i] <= self.eps)[0]
            neighbors.append(neighbor_indices)
            
        labels = np.full(n_points, -1) # 初始全部为噪声 (-1)
        core_points = []
        cluster_id = 0
        
        visited = np.zeros(n_points, dtype=bool)
        
        for i in range(n_points):
            if visited[i]:
                continue
                
            visited[i] = True
            
            # 判断是否为核心点 (包含自身)
            if len(neighbors[i]) >= self.min_samples:
                # 只有 Primary Point 才能发起新的簇，Ghost Point 只能被动加入
                if not is_ghost[i]:
                    core_points.append(point_ids[i])
                    self._expand_cluster(i, neighbors, labels, visited, cluster_id, is_ghost, core_points, point_ids)
                    cluster_id += 1
            else:
                # 标记为噪声 (已经是 -1 了)
                pass
                
        # 构建返回结果
        labels_dict = {point_ids[i]: int(labels[i]) for i in range(n_points)}
        
        # 过滤掉 ghost points 的 core status，我们只关心真正的 core points
        real_core_points = [pid for i, pid in enumerate(point_ids) if pid in core_points and not is_ghost[i]]
        
        return labels_dict, real_core_points
        
    def _expand_cluster(self, point_idx: int, neighbors: List[np.ndarray], labels: np.ndarray, 
                        visited: np.ndarray, cluster_id: int, is_ghost: List[bool], 
                        core_points: List[int], point_ids: List[int]):
        """
        BFS 扩展簇。
        """
        labels[point_idx] = cluster_id
        
        # 使用列表作为队列
        queue = list(neighbors[point_idx])
        
        while queue:
            curr_idx = queue.pop(0)
            
            if not visited[curr_idx]:
                visited[curr_idx] = True
                
                # 如果是核心点，将其邻居加入队列
                if len(neighbors[curr_idx]) >= self.min_samples:
                    if point_ids[curr_idx] not in core_points:
                        core_points.append(point_ids[curr_idx])
                    # 只有当 curr_idx 不是 Ghost Point 时，才将其邻居加入扩展队列
                    # 这样可以防止跨越分区的无限蔓延，保证局部性
                    if not is_ghost[curr_idx]:
                        # 避免重复添加已经在队列中的点
                        for n_idx in neighbors[curr_idx]:
                            if not visited[n_idx]:
                                queue.append(n_idx)
                                
            # 如果该点尚未被分配到任何簇，则将其分配到当前簇
            if labels[curr_idx] == -1:
                labels[curr_idx] = cluster_id
