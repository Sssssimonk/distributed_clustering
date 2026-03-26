import numpy as np
from sklearn.datasets import make_moons, make_blobs
import os

def generate_variable_density_data(n_samples: int = 1000, noise: float = 0.05, random_state: int = 42) -> np.ndarray:
    """
    生成包含不同密度簇和噪声的数据集，用于验证 HDBSCAN 的优越性。
    """
    np.random.seed(random_state)
    
    # 1. 密集的半月形 (Moons)
    moons_data, _ = make_moons(n_samples=n_samples // 3, noise=noise, random_state=random_state)
    moons_data = moons_data * 2.0  # 放大
    
    # 2. 稀疏的高斯斑块 (Blobs)
    blobs_data, _ = make_blobs(n_samples=n_samples // 3, centers=[(5, 5), (-5, -5)], 
                               cluster_std=[1.5, 2.0], random_state=random_state)
                               
    # 3. 极密集的小斑块
    dense_blob, _ = make_blobs(n_samples=n_samples // 6, centers=[(0, -3)], 
                               cluster_std=[0.2], random_state=random_state)
                               
    # 4. 随机背景噪声
    n_noise = n_samples - (n_samples // 3) * 2 - (n_samples // 6)
    noise_data = np.random.uniform(low=-8, high=8, size=(n_noise, 2))
    
    # 合并
    data = np.vstack([moons_data, blobs_data, dense_blob, noise_data])
    
    # 打乱顺序
    np.random.shuffle(data)
    
    return data

def save_data(data: np.ndarray, filepath: str):
    """保存数据到 CSV 文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savetxt(filepath, data, delimiter=",", header="x,y", comments="")
    print(f"Data saved to {filepath}")

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    # 生成小规模测试数据
    small_data = generate_variable_density_data(n_samples=2000)
    save_data(small_data, os.path.join(data_dir, "test_data_2k.csv"))
    
    # 生成中等规模数据用于 Scalability 测试
    med_data = generate_variable_density_data(n_samples=10000)
    save_data(med_data, os.path.join(data_dir, "test_data_10k.csv"))
