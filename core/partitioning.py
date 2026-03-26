import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

class BasePartitioner(ABC):
    """
    空间划分器的抽象基类。
    所有子类必须实现 partition 方法。
    """
    def __init__(self, margin: float):
        """
        :param margin: 边界缓冲距离（DBSCAN中为eps，HDBSCAN中为max_dist）
        """
        self.margin = margin

    @abstractmethod
    def partition(self, point_id: int, coordinates: np.ndarray) -> List[Tuple[int, Dict[str, Any]]]:
        """
        根据点的坐标进行分区，并生成必要的 Ghost Points（边界复制点）。
        
        :param point_id: 原始点的唯一 ID
        :param coordinates: 点的坐标数组
        :return: 返回一个列表，包含 (partition_id, point_metadata)
                 point_metadata 至少包含 {'id': point_id, 'coords': coordinates, 'is_ghost': bool}
        """
        pass

class GridPartitioner(BasePartitioner):
    """
    简单的网格划分器，适用于 DBSCAN。
    """
    def __init__(self, margin: float, cell_size: float):
        super().__init__(margin)
        self.cell_size = cell_size

    def _get_cell_coords(self, coordinates: np.ndarray) -> Tuple[int, ...]:
        return tuple(np.floor(coordinates / self.cell_size).astype(int))

    def partition(self, point_id: int, coordinates: np.ndarray) -> List[Tuple[int, Dict[str, Any]]]:
        primary_cell = self._get_cell_coords(coordinates)
        results = [(hash(primary_cell), {
            'id': point_id, 
            'coords': coordinates, 
            'is_ghost': False,
            'partition_id': hash(primary_cell)
        })]
        
        # 检查是否需要复制到相邻网格 (仅以 2D 为例，可扩展至多维)
        # 为了工程鲁棒性，这里实现一个通用的多维邻居检查
        dims = len(coordinates)
        offsets = [-1, 0, 1]
        import itertools
        
        for offset_tuple in itertools.product(offsets, repeat=dims):
            if all(o == 0 for o in offset_tuple):
                continue
                
            neighbor_cell = tuple(p + o for p, o in zip(primary_cell, offset_tuple))
            
            # 计算点到邻居网格边界的距离
            # 简化起见，如果点距离当前网格某一边界的距离小于 margin，则复制到对应邻居
            is_close = True
            for i in range(dims):
                if offset_tuple[i] == -1:
                    dist_to_boundary = coordinates[i] - primary_cell[i] * self.cell_size
                    if dist_to_boundary > self.margin:
                        is_close = False
                        break
                elif offset_tuple[i] == 1:
                    dist_to_boundary = (primary_cell[i] + 1) * self.cell_size - coordinates[i]
                    if dist_to_boundary > self.margin:
                        is_close = False
                        break
            
            if is_close:
                results.append((hash(neighbor_cell), {
                    'id': point_id,
                    'coords': coordinates,
                    'is_ghost': True,
                    'partition_id': hash(neighbor_cell)
                }))
                
        return results

class KDTreePartitioner(BasePartitioner):
    """
    基于 KD-Tree 的空间划分器，适用于 HDBSCAN，能更好地处理不均匀分布的数据。
    """
    def __init__(self, margin: float, tree_bounds: List[Tuple[np.ndarray, np.ndarray]], leaf_ids: List[int]):
        """
        :param margin: 边界缓冲距离 (max_dist)
        :param tree_bounds: 预先在 Driver 端计算好的各个叶子节点的空间边界 [(min_bound, max_bound), ...]
        :param leaf_ids: 对应的分区 ID 列表
        """
        super().__init__(margin)
        self.tree_bounds = tree_bounds
        self.leaf_ids = leaf_ids

    def partition(self, point_id: int, coordinates: np.ndarray) -> List[Tuple[int, Dict[str, Any]]]:
        results = []
        
        for i, (min_bound, max_bound) in enumerate(self.tree_bounds):
            partition_id = self.leaf_ids[i]
            
            # 判断点是否在当前分区内
            is_inside = np.all(coordinates >= min_bound) and np.all(coordinates < max_bound)
            
            if is_inside:
                results.append((partition_id, {
                    'id': point_id,
                    'coords': coordinates,
                    'is_ghost': False,
                    'partition_id': partition_id
                }))
            else:
                # 判断是否在 margin 范围内 (Ghost Point)
                # 计算点到该矩形边界的最短距离
                dist_to_box = 0.0
                for d in range(len(coordinates)):
                    if coordinates[d] < min_bound[d]:
                        dist_to_box += (min_bound[d] - coordinates[d]) ** 2
                    elif coordinates[d] > max_bound[d]:
                        dist_to_box += (coordinates[d] - max_bound[d]) ** 2
                
                if np.sqrt(dist_to_box) <= self.margin:
                    results.append((partition_id, {
                        'id': point_id,
                        'coords': coordinates,
                        'is_ghost': True,
                        'partition_id': partition_id
                    }))
                    
        return results

def build_kdtree_bounds(sample_points: np.ndarray, max_partitions: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int]]:
    """
    在 Driver 端基于采样数据构建 KD-Tree 边界。
    这是一个简化的实现，通过递归中位数划分空间。
    """
    bounds = []
    
    def split_space(points, min_b, max_b, depth, max_leaves):
        if len(bounds) >= max_leaves or len(points) == 0:
            bounds.append((min_b, max_b))
            return
            
        dim = depth % points.shape[1]
        median_val = np.median(points[:, dim])
        
        left_mask = points[:, dim] < median_val
        right_mask = ~left_mask
        
        # 构造左子树边界
        left_max_b = max_b.copy()
        left_max_b[dim] = median_val
        
        # 构造右子树边界
        right_min_b = min_b.copy()
        right_min_b[dim] = median_val
        
        # 如果还要继续分，且两边都有数据
        if sum(left_mask) > 0 and sum(right_mask) > 0 and len(bounds) + 2 <= max_leaves:
            split_space(points[left_mask], min_b, left_max_b, depth + 1, max_leaves)
            split_space(points[right_mask], right_min_b, max_b, depth + 1, max_leaves)
        else:
            bounds.append((min_b, max_b))

    if len(sample_points) > 0:
        dims = sample_points.shape[1]
        # 初始全局边界稍微放大一点，防止边缘点漏掉
        global_min = np.min(sample_points, axis=0) - 1.0
        global_max = np.max(sample_points, axis=0) + 1.0
        split_space(sample_points, global_min, global_max, 0, max_partitions)
        
    leaf_ids = list(range(len(bounds)))
    return bounds, leaf_ids
