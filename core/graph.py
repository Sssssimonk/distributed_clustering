from typing import List, Tuple, Dict, Set

class UnionFind:
    """
    并查集数据结构，用于合并局部簇。
    """
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, i: int) -> int:
        if i not in self.parent:
            self.parent[i] = i
            self.rank[i] = 0
            return i
            
        if self.parent[i] == i:
            return i
            
        # 路径压缩
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i != root_j:
            # 按秩合并
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1

    def get_components(self) -> Dict[int, List[int]]:
        """
        返回所有连通分量。
        :return: {root_id: [node1, node2, ...]}
        """
        components = {}
        for node in self.parent.keys():
            root = self.find(node)
            if root not in components:
                components[root] = []
            components[root].append(node)
        return components


class KruskalMST:
    """
    Kruskal 算法构建最小生成树。
    适用于 HDBSCAN 中的局部和全局 MST 构建。
    """
    def __init__(self):
        self.uf = UnionFind()
        self.mst_edges = []

    def build(self, edges: List[Tuple[int, int, float]], num_nodes: int = None) -> List[Tuple[int, int, float]]:
        """
        构建最小生成树。
        
        :param edges: 边列表，格式为 [(node1, node2, weight), ...]
        :param num_nodes: 节点总数（用于提前终止优化，可选）
        :return: MST 的边列表
        """
        # 按权重升序排序
        sorted_edges = sorted(edges, key=lambda x: x[2])
        self.mst_edges = []
        
        edges_added = 0
        for u, v, weight in sorted_edges:
            if self.uf.find(u) != self.uf.find(v):
                self.uf.union(u, v)
                self.mst_edges.append((u, v, weight))
                edges_added += 1
                
                # 如果已知节点总数，添加了 N-1 条边后即可提前终止
                if num_nodes is not None and edges_added == num_nodes - 1:
                    break
                    
        return self.mst_edges
