"""
SVM模型测试脚本 - 验证SVM功能用于特征选择评估
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

print("=" * 50)
print("SVM模型测试 - 用于特征选择评估")
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

print(f"    特征矩阵: {X.shape}")
print(f"    标签分布: 0={np.sum(y==0)}, 1={np.sum(y==1)}")

# 3. SVM模型测试
print("\n[3] SVM模型测试...")
# 使用RBF核
svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=42
)

# 5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(svm, X, y, cv=cv, scoring='accuracy')

print(f"    5折交叉验证准确率: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
print(f"    各折准确率: {[f'{s:.4f}' for s in scores]}")

print("\n测试完成!")
