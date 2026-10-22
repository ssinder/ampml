# -*- coding: utf-8 -*-
"""
特征预处理脚本
功能：
1. 方差过滤 (Variance Threshold) - 在标准化前执行
2. 特征标准化 (StandardScaler)
3. Pearson相关性过滤 (去除高相关特征)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "feature_extraction" / "features" / "results"
INPUT_FILE = OUTPUT_DIR / "all_features_combined.csv"

# 读取数据
print("=" * 70)
print("Step 1: Loading Data")
print("=" * 70)
df = pd.read_csv(INPUT_FILE)
print(f"Original data shape: {df.shape}")

# 分离ID和特征
ids = df['ID']
labels = ids.str.extract(r'L(\d+)_')[0].astype(int)
X = df.drop('ID', axis=1)

# 处理缺失值和无穷值
X = X.replace([np.inf, -np.inf], np.nan)
nan_counts = X.isnull().sum().sum()
print(f"NaN values found: {nan_counts}")
if nan_counts > 0:
    X = X.fillna(X.mean())
    print("NaN values filled with column means")

print(f"Original features: {X.shape[1]}")

# ============================================================================
# Step 2: 方差过滤 - 在标准化前执行
# ============================================================================
print("\n" + "=" * 70)
print("Step 2: Variance Threshold Filtering (BEFORE Standardization)")
print("=" * 70)

# 计算原始方差
variances = X.var()
print(f"Raw variance statistics:")
print(f"  Min variance: {variances.min():.6f}")
print(f"  Max variance: {variances.max():.6f}")
print(f"  Mean variance: {variances.mean():.6f}")
print(f"  Median variance: {variances.median():.6f}")

# 方差阈值设置
variance_threshold = 0.01
print(f"\nUsing variance threshold: {variance_threshold}")

# 应用方差过滤
selector = VarianceThreshold(threshold=variance_threshold)
X_filtered = selector.fit_transform(X)

# 获取保留的特征名
selected_features = X.columns[selector.get_support()].tolist()
print(f"Features before variance filtering: {X.shape[1]}")
print(f"Features after variance filtering: {len(selected_features)}")
print(f"Removed features: {X.shape[1] - len(selected_features)}")

# 更新X为过滤后的数据
X = pd.DataFrame(X_filtered, columns=selected_features)

# 保存方差过滤后的原始数据
var_file = os.path.join(OUTPUT_DIR, "features_variance_filtered.csv")
var_df = X.copy()
var_df.insert(0, 'ID', ids)
var_df.to_csv(var_file, index=False)
print(f"  Saved to: {var_file}")

# ============================================================================
# Step 3: 特征标准化
# ============================================================================
print("\n" + "=" * 70)
print("Step 3: Feature Standardization (StandardScaler)")
print("=" * 70)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print(f"Standardization completed")
print(f"  Mean of scaled features (should be ~0): {X_scaled.mean():.6f}")
print(f"  Std of scaled features (should be ~1): {X_scaled.std():.6f}")

# 保存标准化后的数据
scaled_file = os.path.join(OUTPUT_DIR, "features_standardized.csv")
scaled_df = X_scaled_df.copy()
scaled_df.insert(0, 'ID', ids)
scaled_df.to_csv(scaled_file, index=False)
print(f"  Saved to: {scaled_file}")

# ============================================================================
# Step 4: Pearson相关性过滤
# ============================================================================
print("\n" + "=" * 70)
print("Step 4: Pearson Correlation Filtering")
print("=" * 70)

# 计算Pearson相关性矩阵
corr_matrix = X_scaled_df.corr(method='pearson')

# 找出高相关性特征对
correlation_threshold = 0.9
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

print(f"Correlation threshold: {correlation_threshold}")
print(f"High correlation pairs found: {len(high_corr_pairs)}")

# 去除高相关性特征 (保留第一个)
features_to_remove = set()
for feat1, feat2, corr in high_corr_pairs:
    if feat1 not in features_to_remove and feat2 not in features_to_remove:
        features_to_remove.add(feat2)

print(f"Features to remove: {len(features_to_remove)}")

# 保留不相关或低相关特征
features_to_keep = [f for f in X_scaled_df.columns if f not in features_to_remove]
X_final = X_scaled_df[features_to_keep]

print(f"Features after correlation filtering: {len(features_to_keep)}")

# 保存最终结果
final_df = X_final.copy()
final_df.insert(0, 'ID', ids)
final_file = os.path.join(OUTPUT_DIR, "features_processed.csv")
final_df.to_csv(final_file, index=False)
print(f"  Saved to: {final_file}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"Original features: {X.shape[1] + (X.shape[1] - len(selected_features))}")
print(f"After variance filtering: {len(selected_features)}")
print(f"After standardization: {len(selected_features)} (all features)")
print(f"After correlation filtering: {len(features_to_keep)}")

total_original = X.shape[1] + (X.shape[1] - len(selected_features))
total_removed = total_original - len(features_to_keep)
print(f"\nTotal features removed: {total_removed}")
print(f"Final feature count: {len(features_to_keep)}")

# 保存标签文件
labels_file = os.path.join(OUTPUT_DIR, "labels.csv")
labels_df = pd.DataFrame({'ID': ids, 'Label': labels})
labels_df.to_csv(labels_file, index=False)
print(f"\nLabels saved to: {labels_file}")

# 保存特征名列表
features_file = os.path.join(OUTPUT_DIR, "feature_names.txt")
with open(features_file, 'w') as f:
    for feat in features_to_keep:
        f.write(f"{feat}\n")
print(f"Feature names saved to: {features_file}")

print("\n" + "=" * 70)
print("Feature Preprocessing Completed!")
print("=" * 70)
