import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def visualize_clusters(filepath, title):
    df = pd.read_csv(filepath)
    
    plt.figure(figsize=(10, 8))
    
    # 区分噪声和簇
    noise_mask = df['cluster'] == 'NOISE'
    
    # 画噪声点 (灰色, 小点)
    if noise_mask.any():
        plt.scatter(df[noise_mask]['x'], df[noise_mask]['y'], 
                    c='gray', s=10, alpha=0.5, label='Noise')
                    
    # 画聚类点
    clusters = df[~noise_mask]['cluster'].unique()
    palette = sns.color_palette("hsv", len(clusters))
    
    for i, cluster_id in enumerate(clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        plt.scatter(cluster_data['x'], cluster_data['y'], 
                    color=palette[i], s=20, label=f'Cluster {cluster_id}')
                    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 如果簇太多，不显示图例
    if len(clusters) <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
        
    plt.tight_layout()
    
    out_path = filepath.replace(".csv", ".png")
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Clustering Results")
    parser.add_argument("--file", type=str, required=True, help="Path to results CSV")
    parser.add_argument("--title", type=str, default="Clustering Result", help="Plot title")
    
    args = parser.parse_args()
    visualize_clusters(args.file, args.title)
