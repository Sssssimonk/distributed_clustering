import numpy as np
from pyspark.rdd import RDD
from pyspark.sql import SparkSession
from typing import Dict, List, Tuple
from core.spark_utils import get_logger, timeit
from core.partitioning import KDTreePartitioner, build_kdtree_bounds
from core.graph import KruskalMST
from hdbscan.local_graph import LocalHDBSCANGraph
from hdbscan.tree_hierarchy import TreeHierarchy

class DistributedHDBSCAN:
    """
    分布式 HDBSCAN 主调度类。
    """
    def __init__(self, min_samples: int = 5, min_cluster_size: int = 5, max_dist: float = 2.0, max_partitions: int = 16):
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.max_dist = max_dist
        self.max_partitions = max_partitions
        self.logger = get_logger(self.__class__.__name__)

    @timeit()
    def fit(self, rdd: RDD) -> RDD:
        """
        执行分布式 HDBSCAN。
        
        :param rdd: 输入 RDD，格式为 (point_id, coordinates_array)
        :return: 输出 RDD，格式为 (point_id, global_cluster_id)
        """
        self.logger.info(f"Starting Distributed HDBSCAN with min_samples={self.min_samples}, max_dist={self.max_dist}")
        
        # Phase 1: 空间划分
        partitioned_rdd = self._spatial_partitioning(rdd)
        # partitioned_rdd 在 _spatial_partitioning 内部已经被 persist 过了
        
        # Phase 2: 局部 MST 构建
        local_edges_rdd = self._build_local_mst(partitioned_rdd)
        # local_edges_rdd 在 _build_local_mst 内部已经被 persist 过了
        
        # Phase 3: 全局 MST 合并
        global_mst = self._merge_global_mst(local_edges_rdd)
        
        # Phase 4: 树凝缩与簇提取
        cluster_labels = self._extract_clusters(global_mst)
        
        # 分配最终标签
        final_labels_rdd = self._assign_labels(rdd, cluster_labels)
        
        partitioned_rdd.unpersist()
        local_edges_rdd.unpersist()
        
        self.logger.info("Distributed HDBSCAN completed.")
        return final_labels_rdd

    @timeit()
    def _spatial_partitioning(self, rdd: RDD) -> RDD:
        self.logger.info("Phase 1: Spatial Partitioning (KD-Tree)...")
        
        # 1. 采样数据用于构建 KD-Tree 边界
        # 假设数据量很大，采样 1% 或最多 10000 个点
        sample_fraction = min(10000.0 / max(1, rdd.count()), 1.0)
        sample_points = np.array(rdd.sample(False, sample_fraction).map(lambda x: x[1]).collect())
        
        if len(sample_points) == 0:
            raise ValueError("RDD is empty or sample size is 0.")
            
        # 2. 在 Driver 端构建 KD-Tree 边界
        tree_bounds, leaf_ids = build_kdtree_bounds(sample_points, self.max_partitions)
        self.logger.info(f"Built KD-Tree with {len(leaf_ids)} partitions.")
        
        partitioner = KDTreePartitioner(margin=self.max_dist, tree_bounds=tree_bounds, leaf_ids=leaf_ids)
        
        # 3. 广播并分区
        partitioned_rdd = rdd.flatMap(lambda x: partitioner.partition(x[0], x[1]))
        
        # 强制触发行动 (Action) 以精确记录 Phase 1 的真实耗时
        partitioned_rdd.persist()
        num_records = partitioned_rdd.count()
        self.logger.info(f"Phase 1 generated {num_records} partitioned records (including ghosts).")
        
        return partitioned_rdd

    @timeit()
    def _build_local_mst(self, partitioned_rdd: RDD) -> RDD:
        self.logger.info("Phase 2: Building Local MSTs...")
        min_samples = self.min_samples
        
        def run_local_graph(partition_id, points_iterator):
            points_list = list(points_iterator)
            local_builder = LocalHDBSCANGraph(min_samples=min_samples)
            
            local_mst, cross_edges, core_dists = local_builder.build_local_mst(points_list)
            
            # 将局部 MST 边和跨界边合并返回
            all_edges = local_mst + cross_edges
            return all_edges

        # groupByKey 将同一个 partition_id 的数据收集到一起
        local_edges_rdd = partitioned_rdd.groupByKey().flatMap(lambda x: run_local_graph(x[0], x[1]))
        
        # 强制触发行动 (Action) 以精确记录 Phase 2 的真实局部计算耗时
        local_edges_rdd.persist()
        num_edges = local_edges_rdd.count()
        self.logger.info(f"Phase 2 generated {num_edges} compressed local/boundary edges.")
        
        return local_edges_rdd

    @timeit()
    def _merge_global_mst(self, local_edges_rdd: RDD) -> List[Tuple[int, int, float]]:
        self.logger.info("Phase 3: Merging Global MST...")
        
        # 将所有边收集到 Driver 端
        # 得益于局部 MST 压缩，这里的边数应该是 O(N) 级别，而不是 O(N^2)
        all_edges = local_edges_rdd.collect()
        self.logger.info(f"Collected {len(all_edges)} edges for global merge.")
        
        # 在 Driver 端运行 Kruskal 构建全局 MST
        mst_builder = KruskalMST()
        global_mst = mst_builder.build(all_edges)
        
        self.logger.info(f"Global MST built with {len(global_mst)} edges.")
        return global_mst

    @timeit()
    def _extract_clusters(self, global_mst: List[Tuple[int, int, float]]) -> Dict[int, int]:
        self.logger.info("Phase 4: Tree Condensation and Cluster Extraction...")
        
        hierarchy_builder = TreeHierarchy(min_cluster_size=self.min_cluster_size)
        cluster_labels = hierarchy_builder.build_and_extract(global_mst)
        
        self.logger.info(f"Extracted {len(set(cluster_labels.values()))} clusters.")
        return cluster_labels

    @timeit()
    def _assign_labels(self, rdd: RDD, cluster_labels: Dict[int, int]) -> RDD:
        self.logger.info("Assigning final labels to RDD...")
        
        # 获取 SparkContext 以广播字典
        sc = rdd.context
        broadcast_labels = sc.broadcast(cluster_labels)
        
        def map_label(point_tuple):
            pid, coords = point_tuple
            label = broadcast_labels.value.get(pid, -1)
            # 转换为字符串标签，保持与 DBSCAN 输出一致
            final_label = f"Cluster_{label}" if label != -1 else "NOISE"
            return (pid, final_label)
            
        return rdd.map(map_label)
