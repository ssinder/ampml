"""
递归消除特征选择 (RFE) - 测试脚本
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

print("=" * 50)
print("递归消除特征选择 (RFE) - 测试")
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

# 3. RFE特征选择
print("\n[3] 递归消除特征选择...")
# 使用较小的n_estimators加速测试
base_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    random_state=42,
    n_jobs=1
)

# 选择100个特征
n_features_to_select = 100
rfe = RFE(
    estimator=base_model,
    n_features_to_select=n_features_to_select,
    step=10  # 每次消除10个特征
)

print(f"    目标选择特征数: {n_features_to_select}")
rfe.fit(X, y)

# 4. 获取结果
print("\n[4] RFE结果分析...")
selected_mask = rfe.support_
selected_features = feature_names_array[selected_mask]
feature_ranking = rfe.ranking_

# 排名前10的特征
top_10_idx = np.where(feature_ranking <= 10)[0]
print(f"\n    排名前10的特征:")
for i, idx in enumerate(top_10_idx[:10]):
    print(f"        {i+1}. {feature_names_array[idx]} (排名: {feature_ranking[idx]})")

print(f"\n    选择的特征数量: {np.sum(selected_mask)} / {len(feature_names_list)}")
print("\n测试完成!")
