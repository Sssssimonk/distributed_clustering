#!/bin/bash

# =====================================================================
# PySpark 启动脚本
# =====================================================================

# 1. 定义你的 Conda 虚拟环境中的 Python 可执行文件路径
# 如果你之后换了环境，只需要修改这里的路径即可
PYTHON_EXEC="/opt/anaconda3/envs/LSpark/bin/python"

# 2. 导出 Spark 相关的环境变量，强制 Driver 和 Worker 使用相同的 Python
export PYSPARK_PYTHON="$PYTHON_EXEC"
export PYSPARK_DRIVER_PYTHON="$PYTHON_EXEC"

# 3. 运行实验脚本 (将传递给 run.sh 的参数原封不动地传给 run_experiment.py)
# 例如: ./run.sh --algo hdbscan --data data/test_data_2k.csv --cores 4
"$PYTHON_EXEC" scripts/run_experiment.py "$@"
