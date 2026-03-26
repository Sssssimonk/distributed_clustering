import numpy as np
from pyspark.rdd import RDD
from pyspark.sql import SparkSession
from typing import Dict, List, Tuple
from core.spark_utils import get_logger, timeit
from core.partitioning import GridPartitioner
from core.graph import UnionFind
from dbscan.local_dbscan import LocalDBSCAN

class DistributedDBSCAN:
    """
    分布式 DBSCAN 算法的主调度类。
    """
    def __init__(self, eps: float, min_samples: int, cell_size: float = None):
        self.eps = eps
        self.min_samples = min_samples
        # 如果未指定 cell_size，默认使用 2 * eps，保证相邻网格可以覆盖 eps 范围
        self.cell_size = cell_size if cell_size is not None else 2.0 * eps
        self.logger = get_logger(self.__class__.__name__)

    @timeit()
    def fit(self, rdd: RDD) -> RDD:
        """
        执行分布式 DBSCAN。
        
        :param rdd: 输入 RDD，格式为 (point_id, coordinates_array)
        :return: 输出 RDD，格式为 (point_id, global_cluster_id)
        """
        self.logger.info(f"Starting Distributed DBSCAN with eps={self.eps}, min_samples={self.min_samples}")
        
        # Phase 1: 空间划分与 Ghost Point 生成
        partitioned_rdd = self._spatial_partitioning(rdd)
        partitioned_rdd.persist()
        
        # Phase 2: 局部 DBSCAN
        local_results_rdd = self._local_dbscan(partitioned_rdd)
        local_results_rdd.persist()
        
        # Phase 3: 全局簇合并 (Driver 端 Union-Find)
        global_mapping = self._merge_clusters(local_results_rdd)
        
        # Phase 4: 分配最终标签
        final_labels_rdd = self._assign_labels(local_results_rdd, global_mapping)
        
        partitioned_rdd.unpersist()
        local_results_rdd.unpersist()
        
        self.logger.info("Distributed DBSCAN completed.")
        return final_labels_rdd

    @timeit()
    def _spatial_partitioning(self, rdd: RDD) -> RDD:
        self.logger.info("Phase 1: Spatial Partitioning...")
        partitioner = GridPartitioner(margin=self.eps, cell_size=self.cell_size)
        
        # rdd: (point_id, coords)
        # flatMap 输出: [(partition_id, point_dict), ...]
        return rdd.flatMap(lambda x: partitioner.partition(x[0], x[1]))

    @timeit()
    def _local_dbscan(self, partitioned_rdd: RDD) -> RDD:
        self.logger.info("Phase 2: Local DBSCAN...")
        eps = self.eps
        min_samples = self.min_samples
        
        def run_local(partition_id, points_iterator):
            points_list = list(points_iterator)
            local_engine = LocalDBSCAN(eps=eps, min_samples=min_samples)
            labels_dict, core_points = local_engine.fit(points_list)
            
            # 返回格式: (partition_id, (labels_dict, core_points, points_list))
            # 为了全局合并，我们需要保留 points_list 中的 is_ghost 信息
            return [(partition_id, (labels_dict, core_points, points_list))]

        # groupByKey 将同一个 partition_id 的数据收集到一起
        return partitioned_rdd.groupByKey().flatMap(lambda x: run_local(x[0], x[1]))

    @timeit()
    def _merge_clusters(self, local_results_rdd: RDD) -> Dict[str, str]:
        self.logger.info("Phase 3: Global Cluster Merging...")
        
        # 收集所有局部的结果到 Driver 端
        # 注意：在真实超大规模场景下，这里可能需要进一步的分布式合并。
        # 但对于课程 Project，收集边界信息到 Driver 端构建 Union-Find 是合理且高效的。
        local_results = local_results_rdd.collect()
        
        uf = UnionFind()
        
        # 记录每个 point_id 在各个 partition 中的 local_cluster_id
        # 格式: {point_id: [(partition_id, local_cluster_id), ...]}
        point_to_clusters = {}
        
        for partition_id, (labels_dict, core_points, points_list) in local_results:
            for p in points_list:
                pid = p['id']
                cid = labels_dict.get(pid, -1)
                
                # 忽略局部噪声点
                if cid != -1:
                    global_cid = f"{partition_id}_{cid}"
                    if pid not in point_to_clusters:
                        point_to_clusters[pid] = []
                    point_to_clusters[pid].append(global_cid)
                    # 确保节点在并查集中初始化
                    uf.find(global_cid)
                    
        # 如果同一个点在不同分区中都属于某个簇，说明这两个局部的簇应该合并
        for pid, cluster_list in point_to_clusters.items():
            if len(cluster_list) > 1:
                first_cid = cluster_list[0]
                for other_cid in cluster_list[1:]:
                    uf.union(first_cid, other_cid)
                    
        # 构建从 local_cluster_id 到 final_global_cluster_id 的映射
        components = uf.get_components()
        global_mapping = {}
        
        # 为每个连通分量分配一个唯一的全局 ID
        global_id_counter = 0
        for root, members in components.items():
            for member in members:
                global_mapping[member] = f"GlobalCluster_{global_id_counter}"
            global_id_counter += 1
            
        return global_mapping

    @timeit()
    def _assign_labels(self, local_results_rdd: RDD, global_mapping: Dict[str, str]) -> RDD:
        self.logger.info("Phase 4: Assigning Final Labels...")
        
        def map_labels(partition_data):
            partition_id, (labels_dict, core_points, points_list) = partition_data
            results = []
            for p in points_list:
                # 只输出 primary points，丢弃 ghost points，防止重复输出
                if not p['is_ghost']:
                    pid = p['id']
                    cid = labels_dict.get(pid, -1)
                    if cid == -1:
                        final_label = "NOISE"
                    else:
                        local_cid = f"{partition_id}_{cid}"
                        # 如果在映射中找不到（例如孤立的局部簇），则使用其 local_cid 作为全局标识
                        final_label = global_mapping.get(local_cid, local_cid)
                    results.append((pid, final_label))
            return results
            
        return local_results_rdd.flatMap(map_labels)
