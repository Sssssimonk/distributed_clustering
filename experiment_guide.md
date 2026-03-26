# 分布式 HDBSCAN 实验设计与评估指南 (Experiment Guide)

## 0. 导言：为什么要做这些实验？
作为一项分布式算法的“深度作业 (Deep Project)”，仅仅“跑通代码”是不够的。你需要通过精心设计的实验，向教授证明两件事：
1. **你懂机器学习**：你明白为什么我们要费劲去实现 HDBSCAN 而不是普通的 DBSCAN（即算法优越性）。
2. **你懂分布式系统**：你明白 MapReduce 的核心思想，能证明你设计的系统在计算资源（Cores）和数据量（Data Size）增加时具有极佳的**扩展性 (Scalability)**，并且深刻理解系统的串行瓶颈 (Amdahl's Law)。

为此，请严格按照以下三个实验组进行数据收集、作图与报告撰写。

---

## 实验组 1：算法有效性与优越性验证 (Algorithm Correctness & Superiority)

### 1.1 实验目的
证明普通的分布式 DBSCAN 无法处理“变密度 (Variable Density)”数据集，而你实现的分布式 HDBSCAN 可以完美解决这一痛点。

### 1.2 实验操作步骤
1. **准备数据**：使用项目内置的脚本生成二维变密度测试集（包含密集的半月形、稀疏的斑块、极密集的小斑块）。
   ```bash
   # 如果 data 目录下还没有 test_data_2k.csv，运行此命令
   python scripts/generate_data.py
   ```
2. **运行对照组 1：DBSCAN (小 `eps`)**
   ```bash
   ./run.sh --algo dbscan --data data/test_data_2k.csv --cores 4 --eps 0.3
   python scripts/visualize.py --file data/test_data_2k_dbscan_results.csv --title "DBSCAN (eps=0.3)"
   ```
3. **运行对照组 2：DBSCAN (大 `eps`)**
   ```bash
   ./run.sh --algo dbscan --data data/test_data_2k.csv --cores 4 --eps 1.0
   python scripts/visualize.py --file data/test_data_2k_dbscan_results.csv --title "DBSCAN (eps=1.0)"
   ```
4. **运行实验组：HDBSCAN**
   ```bash
   ./run.sh --algo hdbscan --data data/test_data_2k.csv --cores 4
   python scripts/visualize.py --file data/test_data_2k_hdbscan_results.csv --title "HDBSCAN"
   ```

### 1.3 预期结论与报告撰写要点
- 将三张生成的图片并排放在你的报告中。
- 指出 DBSCAN 的“死穴”：小 `eps` 会把稀疏簇当成噪声丢弃；大 `eps` 会把相邻的不同密度簇“粘”成一团。
- 结论：**HDBSCAN 通过树的 Stability 提取，摆脱了对单一距离阈值的依赖，是处理复杂现实数据的必然选择。**

---

## 实验组 2：强扩展性测试 (Strong Scaling - Number of Cores)

### 2.1 实验目的
这是分布式计算课的核心考点：固定总工作量，通过增加机器（或核心数），验证并行计算是否能按比例减少计算时间。你需要证明算法成功地将计算下推，实现了有效解耦。

### 2.2 实验操作步骤
1. **固定数据**：使用中等规模数据集，如 `test_data_10k.csv`。
2. **控制变量测试**：分别在 1, 2, 4（如果机器支持还可以测 8）个核心下运行 HDBSCAN：
   ```bash
   ./run.sh --algo hdbscan --data data/test_data_10k.csv --cores 1
   ./run.sh --algo hdbscan --data data/test_data_10k.csv --cores 2
   ./run.sh --algo hdbscan --data data/test_data_10k.csv --cores 4
   ```
3. **记录关键指标**：仔细阅读控制台输出日志，提取出以下三个核心耗时：
   - `_spatial_partitioning` 耗时 (Phase 1: 空间划分与数据网络传输)
   - `_build_local_mst` 耗时 (Phase 2: 并行计算距离与局部树构建)
   - `_merge_global_mst` 耗时 (Phase 3: Driver端单点串行树合并)

### 2.3 预期结论与报告撰写要点
- **作图**：画一张堆叠柱状图（X轴为 Core 数，Y轴为时间），用三种不同颜色代表 Phase 1, Phase 2, Phase 3。再画一张加速比曲线 ($Speedup = T_1 / T_n$)。
- **结论 1（局部图压缩的伟光正）**：指出 **Phase 2** 的时间随着核心数的增加几乎呈几何级数下降（完美并行 / Embarrassingly Parallel），证明 KD-Tree 空间划分有效解耦了图计算，避免了跨节点计算。
- **结论 2（展现深刻的系统视野）**：重点指出 **Phase 3** 的时间在不同核心数下保持不变（或略有波动）。告诉教授：因为全局合并必须在 Driver 端单点完成，这里受到**阿姆达尔定律 (Amdahl's Law)** 的限制。这展现了你不但会写代码，还能深刻洞察分布式系统的天然瓶颈。

---

## 实验组 3：弱扩展性与数据规模测试 (Data Scalability)

### 3.1 实验目的
证明你在系统设计中引入的“局部 MST (Local MST) 边压缩策略”有效地拯救了系统的内存并打破了原本 $O(N^2)$ 的时间复杂度诅咒。

### 3.2 实验操作步骤
1. **生成多组数据**：修改 `scripts/generate_data.py`，生成 5000, 10000, 20000, 50000 等不同规模的数据。
2. **固定核心数运行**：例如全部使用 `--cores 4` 运行。
   ```bash
   ./run.sh --algo hdbscan --data data/test_data_5k.csv --cores 4
   # 依次类推...
   ```
3. **记录指标**：
   - 记录每次运行的**总时间 (Total Time)**。
   - 观察控制台输出的**局部边数与全局边数**（例如 "Collected 1999 edges for global merge"）。

### 3.3 预期结论与报告撰写要点
- **作图**：画一张折线图（X轴为数据量 $N$，Y轴为运行时间）。
- **理论对比**：在报告中指出，如果是原始的单机 HDBSCAN，构建完全图需要的边数是 $N \times (N-1) / 2$（例如 $N=20000$ 时，边数近 2 亿条），内存早已溢出，时间复杂度是抛物线级别的指数爆炸。
- **结论**：因为你将每个 Partition 内部 $O(M^2)$ 的边压缩成了 $M-1$ 棵树枝。从控制台可以看到，哪怕有几万个点，最终在网络中传输并到达 Driver 端的边仅仅只有 $O(N)$ 级别。时间曲线呈现平缓的增长，完美证明了该分布式架构对大规模数据的卓越处理能力。

---
**附注**：进行上述所有实验时，请确保使用你最新配置好的 `run.sh` 启动，以保证 Spark 运行在正确的 Python 虚拟环境中，避免序列化崩溃。祝你的 Project 拿高分！