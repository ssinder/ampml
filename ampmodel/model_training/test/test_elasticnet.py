"""
ElasticNet特征选择 - 快速测试脚本
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, StratifiedKFold

print("开始测试...")

# 加载数据
features_df = pd.read_csv('D:/trae/paper/input/features/test/featureselect/features_processed.csv')
labels_df = pd.read_csv('D:/trae/paper/input/features/test/featureselect/labels.csv')
feature_names = pd.read_csv('D:/trae/paper/input/features/test/featureselect/feature_names.txt', header=None)

print(f"数据加载完成: {features_df.shape}")

# 数据预处理
X = features_df.iloc[:, 1:].values
y = labels_df['Label'].values

print(f"特征矩阵: {X.shape}")
print(f"标签分布: 0={np.sum(y==0)}, 1={np.sum(y==1)}")

# 测试ElasticNet（使用较小的参数快速测试）
print("\n测试ElasticNet模型...")
en = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42)
en.fit(X, y)

# 查看系数
coef = en.coef_
non_zero = np.sum(coef != 0)
print(f"非零系数数量: {non_zero} / {len(coef)}")

# 快速交叉验证
print("\n进行交叉验证...")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(en, X, y, cv=cv, scoring='accuracy')
print(f"3折交叉验证准确率: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

print("\n测试完成!")
