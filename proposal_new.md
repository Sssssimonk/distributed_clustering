# Deep Project Proposal: 分布式 HDBSCAN 聚类算法的 Spark 实现

## 1. 项目名称

**Distributed HDBSCAN Clustering with Apache Spark** (基于 Apache Spark 的分布式 HDBSCAN 聚类实现)

## 2. 项目背景与简介 (Introduction)

聚类是数据挖掘中无监督学习的核心任务。传统的 DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 算法因其无需预设簇数量、能发现任意形状的簇以及识别噪声的能力，被广泛应用。然而，DBSCAN 存在一个致命的缺陷：**它对全局单一的距离阈值（`eps`）极其敏感，无法有效处理具有不同密度的聚类结构。** 

为了解决这一问题，**HDBSCAN (Hierarchical DBSCAN)** 被提出。HDBSCAN 将 DBSCAN 转化为层次聚类算法，并通过提取稳定性最高的簇来自动适应不同的密度分布。它不仅保留了 DBSCAN 的所有优点，还彻底消除了对 `eps` 参数的依赖，在实际应用中表现出碾压级的优势。

然而，HDBSCAN 的单机计算复杂度极高。它需要全局计算 $k$ 近邻来获取核心距离（Core Distance），计算任意两点间的互达距离（Mutual Reachability Distance, MRD），并在此基础上构建完全图的最小生成树（Minimum Spanning Tree, MST）。当数据量达到百万级别时，单机的内存和计算时间将成为不可逾越的瓶颈。

本项目旨在**挑战从零开始在 PySpark 中实现分布式的 HDBSCAN 算法**。由于 HDBSCAN 包含全局 KNN 和图论（MST）的计算，它比普通的分布式 DBSCAN 更具挑战性。通过“局部 MST 边压缩”与“全局树合并”的创新策略，本项目将展示如何将一个高度依赖全局图结构的复杂算法转化为高效的分布式 MapReduce 计算流，深度契合本课程“深入理解分布式计算原理”的要求。

---

## 3. HDBSCAN 算法核心概念介绍

在设计分布式版本之前，单机 HDBSCAN 的核心步骤如下：

1. **核心距离 (Core Distance)**：对于每个点 $x$，定义 $core\_dist(x)$ 为它到第 $k$ 个最近邻点的距离（$k$ 即 `min_samples` 参数）。
2. **互达距离 (Mutual Reachability Distance, MRD)**：为了让稀疏区域的点远离，定义点 $a$ 和 $b$ 的互达距离为：
   $$MRD(a, b) = \max \{core\_dist(a), core\_dist(b), dist(a, b)\}$$
3. **构建最小生成树 (MST)**：将数据集视为一个完全图，顶点是数据点，边的权重是它们之间的 MRD。使用 Prim 或 Kruskal 算法构建这棵图的最小生成树。
4. **构建层次聚类树 (Single Linkage)**：将 MST 按边权重从大到小依次移除，连通分量不断分裂，形成一棵树。
5. **凝缩树与提取簇 (Tree Condensation & Extraction)**：计算每个簇在不同密度阈值下的“生命周期”（Stability），选择最稳定的连通分量作为最终的簇，其余散落的点即为噪声。

---

## 4. 分布式架构设计思路 (Distributed Design & Architecture)

分布式 HDBSCAN 的最大痛点是**全局 KNN 搜索**和**完全图上的 MST 构建**（边数高达 $O(N^2)$）。为此，本项目设计了以下 **四阶段 (4-Phase) 启发式分布式架构**，核心创新点在于利用局部 MST 大幅压缩图的边数。

### Phase 1: 空间划分与边界软截断 (Spatial Partitioning & Soft Boundary)
- **空间分区**：摒弃简单的网格划分，采用更均衡的 **KD-Tree** 或 **Quad-Tree** 启发式划分，将高维空间切分为多个互不相交的区域（Partition），分配给不同的 Spark Worker。
- **边界复制 (Ghost Points)**：由于 HDBSCAN 没有绝对的 `eps`，我们引入一个合理的超参数 `max_dist`（基于样本抽样估计）。如果一个点距离分区边界小于 `max_dist`，则将其复制到相邻分区。距离超过 `max_dist` 的点对在实际中几乎总是在建树时被当作噪声边砍掉，这种启发式截断是分布式图计算的关键。

### Phase 2: 局部核心距离与局部 MST 构建 (Local MST Construction)
在每个 Partition 内部并行执行（完全解耦）：
- **局部 KNN**：计算每个点在该分区内（包含边界复制点）的 `min_samples` 近邻，得到 `core_dist`。
- **计算 MRD 与 局部 MST**：在分区内部计算点对的 MRD。**核心创新**：每个分区内部独立运行 Kruskal 算法构建 **局部最小生成树 (Local MST)**。
- **降维打击**：原本一个包含 $M$ 个点的分区，有 $O(M^2)$ 条边；构建局部 MST 后，边数被极致压缩为 $M - 1$ 条。这使得后续的网络传输成为可能。

### Phase 3: 跨分区边提取与全局 MST 合并 (Global MST Merging)
- **跨区连通**：提取跨越分区的边界点对（Primary Point 与 Ghost Point 之间）的 MRD，形成边界边集合。
- **全局合并**：将所有分区生成的 **Local MST 边** 和 **跨区边界边** 汇聚。由于边数已经被压缩到了 $O(N)$ 级别，我们可以将这些边 `collect` 到 Driver 端，运行一次轻量级的全局 Kruskal 算法，得到最终的 **全局最小生成树 (Global MST)**。

### Phase 4: 树凝缩与簇提取 (Tree Condensation & Cluster Extraction)
在 Driver 端拿到全局 MST 后：
- 按边权重降序移除边，模拟层次分裂过程。
- 计算每个节点的 Stability，按照 HDBSCAN 的规则提取最终的稳定簇。
- 将最终的聚类标签（Cluster ID 或 Noise）广播（Broadcast）回各个 Worker，完成 RDD 的最终标记。

---

## 5. 核心伪代码 (Pseudocode)

```text
Input: Dataset RDD D, min_samples k, max_dist
Output: RDD with Point_ID and Cluster_Label

// Phase 1: Spatial Partitioning
1. partitions = build_spatial_tree(D.sample(fraction))
2. partitioned_RDD = D.flatMap(point -> assign_to_partitions_with_ghosts(point, partitions, max_dist))

// Phase 2: Local Core Distance & Local MST (Parallel Execution)
3. local_edges_RDD = partitioned_RDD.groupByKey(partition_id).flatMap(partition_data -> {
4.     local_points, ghost_points = split(partition_data)
5.     
6.     // Calculate Core Distances locally
7.     for p in local_points:
8.         p.core_dist = distance_to_kth_neighbor(p, local_points + ghost_points, k)
9.         
10.    // Calculate MRD and Build Local MST
11.    local_graph_edges = calculate_MRD(local_points)
12.    local_mst = kruskal_mst(local_graph_edges)
13.    
14.    // Extract boundary edges crossing partitions
15.    cross_edges = calculate_MRD_between(local_points, ghost_points)
16.    
17.    return local_mst U cross_edges
18. })

// Phase 3: Global MST Merging
19. all_candidate_edges = local_edges_RDD.collect() // Number of edges is now O(N)
20. global_mst = kruskal_mst(all_candidate_edges)

// Phase 4: Cluster Extraction
21. hierarchy_tree = build_single_linkage_tree(global_mst)
22. cluster_labels_map = extract_stable_clusters(hierarchy_tree)
23. broadcast_labels = sc.broadcast(cluster_labels_map)

// Finalize
24. result_RDD = D.map(point -> (point.id, broadcast_labels.value.get(point.id, "NOISE")))
```

---

## 6. 实验设计与结果分析规划 (Experimental Design)

本项目的实验将重点验证算法的**正确性（能否处理变密度集群）**和**分布式架构的可扩展性（Scalability）**。

### 6.1 对照组实验：正确性与优越性验证 (Correctness & Superiority)
- **数据集设计**：人工生成包含明显密度差异的二维/三维数据集（例如：密集的内环圈与稀疏的外环圈、距离很近但密度完全不同的高斯分布斑块）。
- **对照算法设置**：
  1. **单机版 KMeans**：作为 Baseline，证明基于距离的算法无法处理非凸形状。
  2. **分布式基础版 DBSCAN**（设置固定的 `eps`）：证明单一 `eps` 无法同时兼顾密集簇和稀疏簇（要么稀疏簇被当成噪声，要么密集簇连成一片）。
  3. **本项目实现的分布式 HDBSCAN**。
- **预期结果分析**：通过可视化散点图（Scatter Plots），直观展示只有 HDBSCAN 能够完美提取出所有变密度和任意形状的簇，证明本算法在聚类质量上的绝对优越性。

### 6.2 可扩展性实验 (Scalability)
为了验证设计的局部 MST 压缩策略和 Spark 并行化的效率，设计以下实验：

1. **数据规模扩展性测试 (Data Scalability)**：
   - **设置**：固定 Spark Executor 数量（例如 4 个），使用合成数据集，规模分别设置为 $10万, 50万, 100万, 500万$ 个数据点。
   - **指标**：记录算法总运行时间，以及各个 Phase（分区 Shuffle 耗时、局部 MST 计算耗时、全局合并耗时）的占比。
   - **预期**：得益于局部 MST 将 $O(N^2)$ 的图计算降维，运行时间应呈现出可接受的近似线性或 $O(N \log N)$ 的增长曲线，而非指数级爆炸。

2. **计算资源扩展性测试 (Strong Scaling)**：
   - **设置**：固定数据集大小（例如 $100万$ 个点），逐步增加 Spark 集群的 Executor 数量（例如使用 `local[1]`, `local[2]`, `local[4]`, `local[8]` 模拟）。
   - **指标**：绘制**加速比曲线 (Speedup Curve)**（$Speedup = T_{单机} / T_{集群}$）。
   - **预期与分析**：前期随着核数增加，加速比接近线性上升；后期加速比可能放缓。我们将深入分析瓶颈来源（如网络 Shuffle 开销、`max_dist` 边界复制带来的冗余计算、Driver 端全局合并图的单点耗时），展现对分布式系统底层机制的深刻理解。

---

## 7. 潜在优化与总结 (Potential Improvements & Conclusion)

### 潜在优化探讨
在项目实施过程中或最终报告中，我们将讨论以下潜在改进方向：
1. **彻底消除 Driver 瓶颈**：目前的 Phase 3 在 Driver 端进行全局合并，这对于几百万条边是可行的。若面对数亿级数据，可考虑实现分布式的 Borůvka 算法，将全局连通图合并操作完全下推到 RDD 的 Reduce 阶段。
2. **`max_dist` 的自适应截断**：进一步探索如何通过初步的数据采样，动态且自动地为每个空间分区推断一个安全的 `max_dist`，减少不必要的 Ghost Points 复制网络开销。

### 总结
本项目提出并实现了一个基于 Apache Spark 的分布式 HDBSCAN 算法。通过“空间分区 -> 局部 MST 边压缩 -> 全局图合并 -> 层次树提取”的流水线架构，巧妙地解决了 HDBSCAN 算法中全局 KNN 搜索和完全图构建带来的灾难性计算负担。本项目不仅深入贯彻了分布式的底层实现逻辑（全手动实现内部机制，不依赖第三方 ML 库的黑盒方法），同时在算法复杂度和并行化创新上具有极高的深度，是一次兼顾理论先进性与系统工程挑战的深度实践。