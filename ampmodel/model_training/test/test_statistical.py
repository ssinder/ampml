"""
统计检验特征选择 (SelectKBest + f_classif) - 测试脚本
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

print("=" * 50)
print("统计检验特征选择 - 测试")
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

# 3. 统计检验特征选择 (ANOVA F-value)
print("\n[3] 统计检验特征选择 (ANOVA F-value)...")
k_best = 100  # 选择前100个特征
selector = SelectKBest(score_func=f_classif, k=k_best)
selector.fit(X, y)

# 获取分数和选择的特征
scores = selector.scores_
selected_mask = selector.get_support()
selected_features = feature_names_array[selected_mask]
selected_scores = scores[selected_mask]

print(f"    选择特征数量: {k_best}")

# 按分数排序
sorted_idx = np.argsort(selected_scores)[::-1]
print(f"\n    Top 10 重要特征 (按F-score):")
for i in range(min(10, len(sorted_idx))):
    idx = sorted_idx[i]
    print(f"        {i+1}. {selected_features[idx]}: {selected_scores[idx]:.2f}")

# 4. 额外信息
print("\n[4] 分数统计...")
print(f"    最高分数: {np.max(scores):.2f}")
print(f"    最低分数: {np.min(scores):.2f}")
print(f"    平均分数: {np.mean(scores):.2f}")

print("\n测试完成!")
