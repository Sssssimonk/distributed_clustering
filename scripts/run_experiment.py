import argparse
import os
import sys
import numpy as np
import pandas as pd

# 自动将项目根目录添加到 sys.path 中，避免手动 export PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.spark_utils import init_spark

def load_data(spark, filepath):
    """加载 CSV 数据并转换为 (point_id, coords) 的 RDD"""
    df = pd.read_csv(filepath)
    # 转换为列表 [(id, array([x, y])), ...]
    data = [(i, np.array([row['x'], row['y']])) for i, row in df.iterrows()]
    return spark.sparkContext.parallelize(data)

def main():
    parser = argparse.ArgumentParser(description="Run Distributed Clustering Algorithms")
    parser.add_argument("--algo", type=str, choices=["dbscan", "hdbscan"], required=True)
    parser.add_argument("--data", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--cores", type=str, default="*", help="Spark local cores (e.g., 2, 4, *)")
    
    # DBSCAN 参数
    parser.add_argument("--eps", type=float, default=0.5)
    
    # HDBSCAN 参数
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--min_cluster_size", type=int, default=5)
    parser.add_argument("--max_dist", type=float, default=2.0)
    parser.add_argument("--partitions", type=int, default=4)
    
    args = parser.parse_args()
    
    spark = init_spark(app_name=f"Run_{args.algo.upper()}", local_cores=args.cores)
    
    print(f"Loading data from {args.data}...")
    rdd = load_data(spark, args.data)
    print(f"Total points: {rdd.count()}")
    
    if args.algo == "dbscan":
        from dbscan.distributed import DistributedDBSCAN
        model = DistributedDBSCAN(eps=args.eps, min_samples=args.min_samples)
    else:
        from hdbscan.distributed import DistributedHDBSCAN
        model = DistributedHDBSCAN(
            min_samples=args.min_samples, 
            min_cluster_size=args.min_cluster_size,
            max_dist=args.max_dist,
            max_partitions=args.partitions
        )
        
    results_rdd = model.fit(rdd)
    
    # 收集并保存结果
    results = results_rdd.collect()
    
    # 将结果与原始数据合并保存
    df = pd.read_csv(args.data)
    labels_dict = dict(results)
    
    # 保证顺序一致
    df['cluster'] = [labels_dict.get(i, "NOISE") for i in range(len(df))]
    
    out_path = args.data.replace(".csv", f"_{args.algo}_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")
    
    spark.stop()

if __name__ == "__main__":
    main()
