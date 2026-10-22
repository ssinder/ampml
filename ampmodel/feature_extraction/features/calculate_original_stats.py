# -*- coding: utf-8 -*-
"""
计算标准化前特征的均值和标准差
用于后续逆向还原标准化计算过程
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "feature_extraction" / "features" / "results"
INPUT_FILE = OUTPUT_DIR / "all_features_combined.csv"

# 读取数据
print("=" * 70)
print("Loading Data")
print("=" * 70)
df = pd.read_csv(INPUT_FILE)
print(f"Original data shape: {df.shape}")

# 分离ID和特征
ids = df['ID']
X = df.drop('ID', axis=1)

# 处理缺失值和无穷值
X = X.replace([np.inf, -np.inf], np.nan)
nan_counts = X.isnull().sum().sum()
print(f"NaN values found: {nan_counts}")
if nan_counts > 0:
    X = X.fillna(X.mean())
    print("NaN values filled with column means")

print(f"Original features: {X.shape[1]}")

# 应用方差过滤（与preprocess_features.py相同的阈值）
print("\n" + "=" * 70)
print("Applying Variance Threshold Filtering")
print("=" * 70)

variance_threshold = 0.01
print(f"Using variance threshold: {variance_threshold}")

# 计算每个特征的方差
variances = X.var()

# 过滤出方差大于等于阈值的特征
selected_features = variances[variances >= variance_threshold].index.tolist()
X_filtered = X[selected_features]
print(f"Features before variance filtering: {X.shape[1]}")
print(f"Features after variance filtering: {len(selected_features)}")
print(f"Removed features: {X.shape[1] - len(selected_features)}")

# 更新X为过滤后的数据
X = pd.DataFrame(X_filtered, columns=selected_features)

# 计算每个特征的均值和标准差
print("\n" + "=" * 70)
print("Calculating Mean and Std for Each Feature")
print("=" * 70)

feature_stats = []
for feature in X.columns:
    mean_val = X[feature].mean()
    std_val = X[feature].std()
    feature_stats.append({
        'feature': feature,
        'mean': mean_val,
        'std': std_val
    })

# 创建统计信息DataFrame
stats_df = pd.DataFrame(feature_stats)
print(f"Calculated statistics for {len(feature_stats)} features")

# 保存为CSV文件
stats_file = os.path.join(OUTPUT_DIR, "feature_original_stats.csv")
stats_df.to_csv(stats_file, index=False)
print(f"\nStatistics saved to: {stats_file}")

# 显示前几行统计信息
print("\nFirst 10 features statistics:")
print(stats_df.head(10))

print("\n" + "=" * 70)
print("Calculation Completed!")
print("=" * 70)