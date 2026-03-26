import numpy as np
from typing import Tuple

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """计算两个向量之间的欧氏距离。"""
    return np.linalg.norm(p1 - p2)

def manhattan_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """计算两个向量之间的曼哈顿距离。"""
    return np.sum(np.abs(p1 - p2))

def compute_distance_matrix(points: np.ndarray) -> np.ndarray:
    """
    计算一组点之间的两两欧氏距离矩阵。
    使用 scipy 替代双重循环，保证在局部 mapPartitions 中的向量化执行效率。
    """
    from scipy.spatial.distance import pdist, squareform
    if len(points) == 0:
        return np.array([])
    distances = pdist(points, metric='euclidean')
    return squareform(distances)

def is_within_eps(p1: np.ndarray, p2: np.ndarray, eps: float) -> bool:
    """判断两点之间距离是否在 eps 范围内。"""
    return euclidean_distance(p1, p2) <= eps
