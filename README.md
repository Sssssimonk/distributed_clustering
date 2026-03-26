# Distributed HDBSCAN & DBSCAN with Apache Spark

This project provides a from-scratch distributed implementation of the **HDBSCAN** and **DBSCAN** clustering algorithms using Apache Spark (PySpark). It is designed to overcome the memory and computational bottlenecks of single-machine clustering by intelligently partitioning spatial data and compressing graph representations.

The implementation strictly follows MapReduce principles without relying on high-level black-box ML libraries, offering deep insights into distributed graph algorithms and density-based clustering.

## Core Features
- **4-Phase Architecture**: Decomposes complex clustering into Spatial Partitioning, Local Graph Construction, Global Merging, and Tree Condensation.
- **Extreme Graph Compression**: Uses Local Minimum Spanning Trees (MST) to compress the typical $O(N^2)$ distance edges down to $O(N)$ before global merging, preventing network and memory overload.
- **KD-Tree Soft Partitioning**: Employs heuristic KD-Tree partitioning with "Ghost Points" (soft boundaries) to effectively handle unevenly distributed data across Spark partitions.
- **Advanced Tree Hierarchy**: Fully manual implementation of HDBSCAN's Single Linkage and cluster stability condensation algorithms.
- **Experiment Ready**: Built-in scripts for generating variable-density datasets, measuring phase-specific runtime (for Amdahl's Law analysis), and visualizing results.

## Project Structure

- `core/`: Common utilities shared between algorithms (vectorized distance metrics, spatial partitioning, Union-Find, Kruskal's MST, and Spark `@timeit` loggers).
- `dbscan/`: Distributed DBSCAN baseline implementation using Grid Partitioning and Union-Find merging.
- `hdbscan/`: Distributed HDBSCAN core implementation.
- `scripts/`: Python scripts for generating datasets, executing batch experiments, and rendering 2D scatter plots.
- `technical_design.md`: In-depth documentation explaining the algorithmic design and distributed philosophy.
- `experiment_guide.md`: Step-by-step instructions for running scalability and correctness experiments.

## Setup

1. **Install dependencies:**
   Ensure you have a Python environment (Conda is recommended) and Java 8+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Spark Environment:**
   If you are using a virtual environment (like Conda), make sure to configure `run.sh` to point to your specific Python executable to avoid `PYTHON_VERSION_MISMATCH` errors between Spark's Driver and Workers.
   Edit `run.sh` and update:
   ```bash
   PYTHON_EXEC="/path/to/your/conda/env/bin/python"
   ```

## Running Experiments

### 1. Generate Synthetic Data
Generates 2D variable-density datasets (Moons, Sparse Blobs, Dense Blobs).
```bash
python scripts/generate_data.py
```
*(This will populate the `data/` folder with CSVs of varying sizes.)*

### 2. Run Distributed Clustering
Use the included `run.sh` script to launch the PySpark job safely.
```bash
# Run HDBSCAN with 4 parallel cores
./run.sh --algo hdbscan --data data/test_data_2k.csv --cores 4

# Run DBSCAN baseline for comparison
./run.sh --algo dbscan --data data/test_data_2k.csv --cores 4 --eps 0.5
```

### 3. Visualize Results
```bash
python scripts/visualize.py --file data/test_data_2k_hdbscan_results.csv --title "Distributed HDBSCAN Result"
```
*(The plot will be saved as a PNG in the same directory as the data.)*

## Documentation
Please refer to:
- **`technical_design.md`** for architectural details and codebase design.
- **`experiment_guide.md`** for instructions on how to evaluate the system's strong/weak scalability and clustering superiority.