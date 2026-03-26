# 分布式 HDBSCAN 技术设计文档 (Technical Design Documentation)

## 1. 架构概览与设计哲学

本项目的核心目标是使用 Apache Spark (PySpark) 从零开始实现 **分布式 HDBSCAN 算法**。HDBSCAN 因其处理变密度聚类和无需设定全局距离阈值 (`eps`) 的优异特性，在聚类质量上碾压传统的 DBSCAN。

然而，HDBSCAN 算法中包含了**全局 K-最近邻 (KNN) 计算**、**完全图互达距离 (MRD) 矩阵**以及**最小生成树 (MST)** 的构建。在单机环境下，其时间与空间复杂度高达 $O(N^2)$。当数据量增加时，单机的内存和计算时间将成为不可逾越的瓶颈。

### 1.1 设计哲学：MapReduce 下的局部图压缩
本架构的最核心创新在于：**通过空间划分将全局图拆分为多个局部子图，利用“局部最小生成树 (Local MST)”极致压缩边数，再将压缩后的边汇聚到 Driver 端进行全局合并与树凝缩。**
这是一种典型的 “Push-down” 思想——将尽可能多的计算（距离计算、KNN、局部树构建）下推到相互独立的 Worker 节点上并行执行，从而避免了跨节点的笛卡尔积级的数据 Shuffle。

整个分布式 HDBSCAN 的执行流程被严格划分为 **4 个流水线阶段 (4-Phase Architecture)**：
1. **Phase 1: 基于 KD-Tree 的空间划分与边界软截断 (Spatial Partitioning)**
2. **Phase 2: 局部核心距离计算与局部 MST 构建 (Local Graph Construction)**
3. **Phase 3: 跨分区边提取与全局 MST 合并 (Global MST Merging)**
4. **Phase 4: 层次树凝缩与稳定簇提取 (Tree Condensation & Extraction)**

---

## 2. 核心模块详细设计

### 2.1 Phase 1: 空间划分与边界软截断 (`core/partitioning.py`)

为了保证后续 Phase 2 的局部计算能够高度并行，我们需要将数据打散到不同的 Spark Partition 中。普通的网格划分 (Grid Partitioning) 容易导致数据倾斜，因此我们采用了基于 **KD-Tree** 的启发式空间划分。

**工程实现逻辑：**
1. **Driver 端采样建树**：由于数据集可能极大，Driver 端通过 `rdd.sample()` 获取少量样本（如 10000 个点）。基于这些样本，递归计算中位数，划分出 $P$ 个互不相交的高维矩形空间（对应叶子节点），生成 KD-Tree 的空间边界。
2. **Worker 端映射分区**：将边界规则广播给 Worker。使用 `flatMap` 算子，遍历每一条数据，判断其落入哪个叶子节点的空间内，并打上 `partition_id`。
3. **边界软截断 (Soft Boundary via Ghost Points)**：
   - 聚类算法的核心痛点是如何处理处于两个分区交界处的点。在分布式 DBSCAN 中，由于有确定的 `eps`，只要把距离边界小于 `eps` 的点复制一份即可。
   - HDBSCAN **没有 `eps`**。为了解决这个问题，我们引入了超参数 `max_dist`。
   - 当一个点（Primary Point）距离某个相邻分区的边界小于 `max_dist` 时，它将被复制一份（标记为 `is_ghost=True`）发送给该相邻分区。这些 Ghost Points 仅用于辅助相邻分区的连通性计算，不作为该分区的主体点。这种启发式的软截断极大地减少了不必要的跨分区图计算。

### 2.2 Phase 2: 局部 MST 构建 (`hdbscan/local_graph.py`) - 【核心并行层】

这一阶段是整个架构扩展性的基石。通过 `groupByKey`，每个 Partition 将分配到属于自己的 Primary Points 以及相邻分区漫游过来的 Ghost Points。

**工程实现逻辑：**
在每个 Worker 内部独立、串行地执行以下逻辑（由于各个分区的数据规模被极大缩减，单机内存足以支撑）：
1. **向量化距离矩阵**：利用 `scipy.spatial.distance` 高效计算当前分区内所有点（包含 Ghost）的成对欧氏距离矩阵。
2. **核心距离计算 (Core Distance)**：对矩阵每行使用 `np.partition` 获取第 `min_samples` 近的距离，得到每个点的 $core\_dist$。
3. **互达距离计算 (Mutual Reachability Distance, MRD)**：
   - $MRD(A, B) = \max \{core\_dist(A), core\_dist(B), dist(A, B)\}$
   - 这里通过 NumPy 的 `np.maximum` 实现了矩阵级别的并发广播操作，效率极高。
4. **局部图压缩 (Local MST)**：
   - 过滤掉两端都是 Ghost Point 的无效边。
   - 如果两端都是 Primary Point，则将其作为局部边，并使用 Kruskal 算法构建**局部最小生成树 (Local MST)**。
   - 这一步将当前分区原本 $O(M^2)$ 条边极致压缩到了 $M-1$ 条（$M$ 为分区内 Primary 点的数量）。
5. **跨界边收集**：如果边的一端是 Primary，另一端是 Ghost，说明这是跨越分区的连接线，我们保留这些边但不参与局部 MST 的构建。
6. **输出**：返回被压缩后的局部 MST 边和跨界边。

### 2.3 Phase 3: 全局 MST 合并 (`hdbscan/distributed.py`)

**工程实现逻辑：**
经过 Phase 2 的降维打击，整个 Spark 集群中残存的边数已经从 $O(N^2)$ 降低到了接近 $O(N)$（$N$ 为总数据量）。
1. 使用 `collect()` 将这少量的边拉取到 Driver 节点的内存中。
2. 在 Driver 端，再次使用 Kruskal 算法，将这些分散的局部树枝和跨区跨界边拼接成一棵**全局最小生成树 (Global MST)**。
3. *注：在亿级数据的工业场景下，这一步可以替换为分布式的 Borůvka 算法继续下推到 MapReduce 中执行，但在百万级以内的数据中，Driver 端聚合是性能最优的工程妥协。*

### 2.4 Phase 4: 树凝缩与最优簇提取 (`hdbscan/tree_hierarchy.py`)

拥有了全局 MST 后，我们进入标准的 HDBSCAN 无参提取流程（Tree Condensation）。

**工程实现逻辑：**
这一模块完全手写，不依赖 `scikit-learn` 或 `scipy` 的高级聚类黑盒，严格复刻了 HDBSCAN 论文中的数学逻辑：
1. **Single Linkage 聚类树**：根据边权从小到大，使用并查集记录每个节点的合并过程（诞生距离、合并规模等），构建一棵二叉树。
2. **凝缩树 (Condensed Tree)**：
   - 自顶向下（权重大到小）遍历二叉树。
   - 当一个簇分裂时，检查两个子簇的大小是否大于 `min_cluster_size`。
   - 如果都大于，视为**真正的簇分裂**；如果有一个小于，视为**噪声脱落 (Noise falling out)**。
   - 噪声脱落不产生新簇，只会导致父簇规模缩小，同时将该距离下的生命周期累加给存活的簇。
3. **计算 Stability**：
   - 对于每个真正的簇，计算其生命周期内的稳定性总和：$Stability = \sum (\lambda_{death} - \lambda_{birth}) \times size$（其中 $\lambda = 1/distance$）。
4. **最优簇提取 (Cluster Extraction)**：
   - 自底向上遍历，如果一个簇自身的 Stability 大于其所有子簇的 Stability 之和，则保留当前簇（不再细分）；否则，将自身 Stability 更新为子簇之和，继续保留子簇。
5. **标签广播**：最终选定簇后，将叶子节点映射到所属的有效簇 ID。将这本字典广播（Broadcast）回 Spark 的 Worker，让 RDD 中的每条记录完成终态映射（标记为特定 Cluster 或 NOISE）。

---

## 3. 代码结构与模块化设计 (Modularity)

为了保证代码的鲁棒性（Robustness）与高工程化标准（DRY 原则），项目剥离了专门的 `core` 组件：

```text
├── core/                      # 算法底层公共组件包
│   ├── distance.py            # 向量化的距离计算，避免使用低效的 for 循环
│   ├── partitioning.py        # 包含抽象基类 BasePartitioner，派生出网格与 KDTree 划分器
│   ├── graph.py               # 原生的 Union-Find 和 Kruskal 算法实现，用于各个 Phase 复用
│   └── spark_utils.py         # 包含强大的 @timeit 装饰器，用于无侵入式地监控各个 Phase 耗时
├── dbscan/                    # 分布式 DBSCAN 对照组实现
├── hdbscan/                   # 核心分布式 HDBSCAN 引擎 (封装为类似于 sklearn 的 .fit() 接口)
└── scripts/                   # 包含数据生成、批量执行与可视化的用户级入口
```

在调度类 `DistributedHDBSCAN` 中，主逻辑被写成了高度语义化的 Pipeline：
```python
partitioned_rdd = self._spatial_partitioning(rdd)
local_edges_rdd = self._build_local_mst(partitioned_rdd)
global_mst = self._merge_global_mst(local_edges_rdd)
cluster_labels = self._extract_clusters(global_mst)
final_labels_rdd = self._assign_labels(rdd, cluster_labels)
```
并在核心 RDD 操作之间合理插入了 `rdd.persist()` 和 `rdd.count()`，**以此强制打断 Spark 的 Lazy Evaluation 机制，确保日志输出的时间是每一阶段实打实的计算耗时，为撰写极高质量的性能扩展性报告打下了坚实基础。**