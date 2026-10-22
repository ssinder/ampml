"""
随机森林特征选择 - 测试脚本
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("=" * 50)
print("随机森林特征选择 - 测试")
print("=" * 50)

# 1. 加载数据
print("\n[1] 加载数据...")
features_df = pd.read_csv('D:/trae/paper/input/features/test/featureselect/features_processed.csv')
labels_df = pd.read_csv('D:/trae/paper/input/features/test/featureselect/labels.csv')
feature_names = pd.read_csv('D:/trae/paper/input/features/test/featureselect/feature_names.txt', header=None)

feature_names_list = feature_names[0].tolist()
print(f"    数据形状: {features_df.shape}")

# 2. 数据预处理
print("\n[2] 数据预处理...")
X = features_df.iloc[:, 1:].values
y = labels_df['Label'].values
feature_names_array = np.array(feature_names_list)

print(f"    特征矩阵: {X.shape}")
print(f"    标签分布: 0={np.sum(y==0)}, 1={np.sum(y==1)}")

# 3. 随机森林特征选择
print("\n[3] 随机森林特征选择...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=1
)
rf.fit(X, y)

# 获取特征重要性
importances = rf.feature_importances_
print(f"    模型训练完成")

# 4. 选择重要特征
print("\n[4] 特征重要性分析...")
# 使用均值作为阈值
threshold = np.mean(importances)
selected_mask = importances >= threshold
selected_features = feature_names_array[selected_mask]
selected_importances = importances[selected_mask]

print(f"    阈值 (均值): {threshold:.6f}")
print(f"    选择的特征数量: {np.sum(selected_mask)} / {len(feature_names_list)}")

# 按重要性排序
sorted_idx = np.argsort(selected_importances)[::-1]
print(f"\n    Top 10 重要特征:")
for i in range(min(10, len(sorted_idx))):
    idx = sorted_idx[i]
    print(f"        {i+1}. {selected_features[idx]}: {selected_importances[idx]:.4f}")

print("\n测试完成!")
