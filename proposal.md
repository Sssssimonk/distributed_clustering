# Proposal

## 1. Project Title

**Distributed DBSCAN Clustering with Apache Spark**

## 2. Description of the Problem

### 2.1 Background

Clustering is a fundamental task in data mining and machine learning. Its goal is to group similar data points together while separating dissimilar ones. Among many clustering algorithms, **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) is especially attractive because it has several useful properties:

- it does not require the number of clusters to be specified in advance;
- it can discover clusters of arbitrary shape;
- it can explicitly identify noise and outliers.

These properties make DBSCAN suitable for many real-world applications such as spatial data analysis, anomaly detection, trajectory mining, and image or embedding-based clustering.

However, despite its usefulness, DBSCAN is difficult to scale to large datasets. The main computational bottleneck comes from **neighborhood queries**: for each point, the algorithm needs to determine which other points lie within a distance threshold `eps`. In a naive implementation, this may require comparing each point with all other points, resulting in quadratic time complexity. Even with indexing structures, the algorithm remains challenging when the dataset becomes large.

### 2.2 Why Distributed DBSCAN?

A sequential DBSCAN implementation works well on a single machine for moderate data sizes, but it becomes inefficient or even infeasible when:

- the dataset is too large to fit comfortably in memory,
- neighborhood searches become expensive,
- cluster expansion needs to be repeated over many points.

Therefore, there is strong motivation to implement DBSCAN in a **distributed computing framework** such as Spark.

Spark is suitable for this task because it supports:

- data partitioning across workers,
- parallel execution of map/reduce style transformations,
- iterative computations,
- in-memory processing for better performance than pure disk-based MapReduce.

### 2.3 Main Challenge

The main difficulty is that **DBSCAN is not naturally parallel**.

Unlike algorithms such as K-means, DBSCAN relies on local density connectivity, which creates several challenges in a distributed setting:

1. **Neighborhood search across partitions**
   A point’s neighbors may lie in another partition, not only in its local partition.
2. **Boundary handling**
   Clusters may cross partition boundaries. If each partition is processed independently, one global cluster may be incorrectly split into several local clusters.
3. **Cluster merging**
   After local clustering, clusters found in different partitions must be merged correctly if they are density-connected through shared or boundary points.

Because of these issues, implementing DBSCAN in Spark is not a matter of simply calling the standard algorithm inside each worker. A proper distributed design is required.

### 2.4 Problem Statement

This project aims to design and implement a **distributed version of DBSCAN in PySpark**, without relying on built-in clustering libraries, in accordance with the course requirement that the internal algorithmic logic must be implemented manually rather than delegated to a library.

The project will focus on the following problem:

> Given a large set of multidimensional data points, design a Spark-based distributed DBSCAN algorithm that can:
>
> - perform clustering in parallel,
> - preserve the semantics of density-based clustering,
> - handle clusters spanning multiple partitions,
> - identify noise points,
> - and demonstrate scalability as data size and computational resources increase.

## 3. Description of the Algorithm

### 3.1 Standard DBSCAN Overview

DBSCAN relies on two parameters:

- `eps`: the radius of the neighborhood around a point;
- `minPts`: the minimum number of points required to form a dense region.

For each point `p`, we define its `eps`-neighborhood as all points within distance `eps` from `p`.

A point is classified as:

- **Core point**: if its `eps`-neighborhood contains at least `minPts` points;
- **Border point**: if it is not a core point but lies within the neighborhood of a core point;
- **Noise point**: if it is neither core nor border.

Clusters are formed by connecting core points that are density-reachable, and attaching border points to nearby core-based clusters.

### 3.2 Sequential DBSCAN Procedure

The standard sequential procedure is:

1. Mark all points as unvisited.
2. For each unvisited point `p`:
   - find all neighbors of `p` within distance `eps`;
   - if the number of neighbors is less than `minPts`, mark `p` as noise temporarily;
   - otherwise, create a new cluster and recursively expand it by visiting all density-reachable neighbors.
3. Continue until all points are processed.

### 3.3 Why Sequential DBSCAN is Hard to Parallelize

The expensive and difficult part is not only the neighborhood query itself, but also the recursive cluster expansion. Once a point is found to be a core point, its neighbors must be explored, then neighbors of those neighbors, and so on. This creates a connected-component style dependency that may cross partition boundaries.

Therefore, a distributed design must solve two problems simultaneously:

- how to **parallelize local clustering work**,
- how to **reconstruct global clusters** after local work is finished.

### 3.4 Proposed Distributed DBSCAN Strategy

This project proposes a **partition-based distributed DBSCAN** with three main phases:

1. **Spatial partitioning of the dataset**
2. **Local DBSCAN within each partition**
3. **Global merging of boundary-connected local clusters**

The design is intentionally practical and course-project oriented: it preserves the spirit of DBSCAN while keeping the implementation manageable in Spark.

------

#### Phase 1: Spatial Partitioning

The dataset will first be partitioned according to spatial location.

For low-dimensional numerical data (especially 2D or moderate dimensions), the feature space can be divided into **grid cells**. Each point is assigned to one primary cell based on its coordinates.

The purpose of spatial partitioning is:

- to reduce the search space for local neighborhood queries;
- to ensure points that are close in space are likely to be processed together;
- to make the algorithm parallelizable.

However, a point near the edge of a cell may have neighbors in adjacent cells. Therefore, naive partitioning would break the correctness of DBSCAN.

To address this, the project will use **boundary replication**:

- if a point lies within distance `eps` from a cell boundary, it is replicated into neighboring cells that may contain its neighbors.

This ensures that local clustering inside a partition has enough information to correctly identify dense connectivity near partition boundaries.

##### Key idea:

A point may appear in multiple partitions, but only one copy is its primary ownership record. Replicated copies are used only to support correct local clustering and later merging.

------

#### Phase 2: Local DBSCAN in Each Partition

After partitioning and replication, each Spark partition (or cell group) will independently run a **local DBSCAN** on the set of points assigned to it.

Inside one partition:

- distances are computed only among points within that partition’s local dataset;
- core/border/noise classification is determined locally;
- local cluster identifiers are generated, for example as `(partition_id, local_cluster_id)`.

This phase is embarrassingly parallel once partition contents are prepared.

The output of local DBSCAN for each point will include:

- point ID,
- coordinates,
- partition ID,
- local cluster ID,
- point type (core, border, noise),
- whether the point is replicated or primary.

At this stage, the same global cluster may still appear as multiple local clusters in different partitions.

------

#### Phase 3: Merging Local Clusters Across Partitions

The final phase merges local clusters that should belong to the same global cluster.

Two local clusters should be merged if they are connected through shared or boundary points. In practice, this can be detected using replicated points:

- if the same original point appears in multiple partitions and is assigned to different local clusters,
- or if two local clusters contain points that are within `eps` across a boundary,

then these local clusters represent the same global DBSCAN cluster and should be merged.

This merging can be formulated as a graph problem:

- each local cluster is treated as a node;
- an edge is added between two local clusters if they should be connected;
- connected components of this graph represent final global clusters.

A simple and effective way to implement this is to use **Union-Find (Disjoint Set Union)** or an equivalent connected-components merging logic.

This phase is critical because it restores the global density connectivity that was temporarily broken by partitioning.

------

### 3.5 High-Level Pseudocode

Below is the verbal pseudocode of the proposed algorithm:

```text
Input:
    Dataset D
    Distance threshold eps
    Density threshold minPts

Output:
    Cluster label for each point

1. Partition the data space into grid cells.
2. For each point p in D:
       assign p to its primary cell;
       if p is within eps of a cell boundary,
           replicate p to neighboring relevant cells.
3. Group points by partition/cell.
4. In parallel, for each partition:
       run local DBSCAN on points in that partition;
       assign local cluster IDs.
5. Collect boundary/replicated point information.
6. Determine which local clusters across partitions should be merged.
7. Build a cluster-merge graph or Union-Find structure.
8. Compute connected components of local clusters.
9. Map each local cluster ID to a final global cluster ID.
10. Output final labels for all original points.
```

------

### 3.6 Expected Advantages of This Design

This design has several advantages:

1. **Parallel local work**
   Most clustering work is done independently inside partitions.
2. **Reduced search space**
   Neighborhood queries are restricted to partition-local data plus replicated boundary points.
3. **Correct boundary handling**
   Replication prevents many cross-partition neighbor relationships from being missed.
4. **Clear decomposition into Spark-friendly steps**
   The algorithm naturally maps to transformations such as `map`, `flatMap`, `groupByKey`/`reduceByKey`, and post-processing for merging.

------

### 3.7 Limitations and Trade-offs

This design also introduces trade-offs:

1. **Replication overhead**
   Boundary replication increases memory and communication cost.
2. **Partition quality matters**
   If partitions are poorly chosen, some cells may be overloaded while others remain sparse.
3. **High-dimensional data is harder**
   Grid partitioning becomes less efficient as dimensionality increases.
4. **Merging complexity**
   The correctness of the final result depends on accurate merge detection.

These limitations are acceptable for a course project and also provide useful opportunities for analysis and discussion in the final report.

------

## 4. Brief Plan for Spark Implementation

### 4.1 Implementation Language and Platform

The project will use **PySpark**, since it satisfies the course requirement of implementing the algorithm with Spark’s Python API.

The implementation will run in Spark local mode during development, and can be evaluated with different numbers of local worker threads to simulate varying parallel resources.

### 4.2 Data Representation

Each input point will be represented as a record such as:

```text
(point_id, coordinates)
```

During processing, additional metadata will be attached:

```text
(point_id, coordinates, primary_partition, replica_partition, flag)
```

and later:

```text
(point_id, coordinates, partition_id, local_cluster_id, point_type)
```

Unique point IDs are important because the same point may appear in multiple partitions due to replication.

### 4.3 Spark Pipeline

The implementation is planned as the following Spark pipeline:

#### Step 1: Load data

- Read a synthetic or real dataset into an RDD/DataFrame.
- Assign each point a unique ID.

#### Step 2: Compute partition assignment

- Use a spatial partitioning function to map each point to a grid cell.
- Determine whether the point needs replication to neighboring cells.

Possible Spark operation:

- `flatMap(point -> [(cell_id, point_with_metadata), ...])`

#### Step 3: Group by partition

- Group all records belonging to the same partition.

Possible Spark operation:

- `groupByKey()` or a more efficient alternative depending on structure.

#### Step 4: Local DBSCAN

- For each partition, run a custom local DBSCAN implementation.
- Produce local cluster labels.

Possible Spark operation:

- `mapValues(local_dbscan)` or `mapPartitions(...)`

#### Step 5: Extract merge candidates

- Identify cases where the same original point or neighboring boundary points connect different local clusters.
- Emit cluster-pair merge relations.

Possible Spark operation:

- `flatMap(...)` to output `(local_cluster_A, local_cluster_B)` pairs.

#### Step 6: Compute global cluster merging

- Apply Union-Find logic or iterative connected-component merging.
- Create a mapping from local cluster IDs to final global cluster IDs.

#### Step 7: Produce final labels

- Join point-level local labels with the global merge mapping.
- Remove duplicate replicated records and keep only original points in the final output.

------

### 4.4 Local DBSCAN Module

The local DBSCAN module will be manually implemented and not replaced by an external clustering library, in compliance with the project requirement.

This module will:

- compute pairwise neighborhood relations inside one partition,
- determine core/border/noise points,
- expand local clusters using BFS/queue-based traversal,
- return a label for each point in the partition.

For simplicity and transparency, the initial version may use direct distance computation within a partition. If time permits, partition-local optimizations may be explored.

### 4.5 Partitioning Design

The initial implementation will use a **uniform grid partitioning** strategy because:

- it is easy to explain,
- easy to implement,
- and closely aligned with the geometry of radius-based neighborhood queries.

The grid size will be chosen in relation to `eps`. A natural starting idea is to make cell width comparable to `eps`, so that only a limited number of neighboring cells need to be considered for replication.

This design may later be discussed or improved in terms of:

- partition skew,
- replication rate,
- sensitivity to data distribution.

### 4.6 Cluster Merging Design

The merging module will treat each local cluster as a temporary identifier. Merge candidates will be generated from partition boundary evidence.

A simple design is:

- represent each local cluster as a node,
- build edges between local clusters that share replicated points or satisfy cross-boundary connectivity,
- compute final connected components.

This can be implemented either:

- partly in Spark,
- or by collecting only the compact merge graph to the driver if its size is manageable.

For a course project, this is a reasonable engineering trade-off because the merge graph is usually much smaller than the full dataset.

### 4.7 Experimental Plan

The project will evaluate the implementation from both **correctness** and **scalability** perspectives.

#### Correctness evaluation

- compare results on small datasets with a sequential DBSCAN implementation;
- verify cluster assignments visually on 2D synthetic datasets if applicable;
- check whether boundary-crossing clusters are correctly merged.

#### Scalability evaluation

The course explicitly requires experiments demonstrating scalability with increasing data size and number of executors in the final presentation/report.
Therefore, the following experiments are planned:

1. **Varying dataset size**
   - e.g. 10K, 50K, 100K, 500K points
2. **Varying number of workers / local cores**
   - e.g. `local[1]`, `local[2]`, `local[4]`, `local[*]`
3. **Possibly varying partition granularity**
   - to study the effect of partition count and replication overhead

Metrics to report:

- total runtime,
- runtime of major stages,
- speedup,
- replication overhead,
- number of merge operations.

### 4.8 Possible Optimizations

If time permits, the project may include one or more optimizations:

- improved partition sizing,
- using `mapPartitions` instead of less efficient grouping patterns,
- caching intermediate RDDs,
- pruning unnecessary replication,
- more efficient local neighborhood search.

These optimizations will be discussed in the presentation/report as potential improvements, which is also encouraged by the project specification.

------

## 5. Expected Contributions

This project is expected to contribute:

1. A complete manual implementation of distributed DBSCAN in Spark;
2. A clear demonstration of how a density-based clustering algorithm can be adapted to a distributed environment;
3. An analysis of the trade-offs among correctness, replication cost, and scalability;
4. Experimental evidence on how the implementation behaves with larger data and more parallelism.

---

## 6. Conclusion

This project proposes a distributed implementation of DBSCAN using Apache Spark. The key idea is to decompose the algorithm into:

- spatial partitioning,
- independent local clustering,
- and global merging of boundary-connected local clusters.

The project is meaningful because DBSCAN is an important clustering algorithm whose density-based nature makes distributed implementation non-trivial. By implementing it manually in Spark, the project will demonstrate both algorithmic understanding and distributed systems thinking, which matches the main goal of the deep project assignment.