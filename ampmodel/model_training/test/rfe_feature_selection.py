"""
递归消除特征选择脚本 (RFE)
用于从1098个特征中选择重要特征，为抗菌肽分类模型做准备
使用随机森林作为基模型进行递归特征消除
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("递归消除特征选择 (RFE) 分析")
print("=" * 60)

# 1. 加载数据
print("\n[1] 加载数据...")
input_dir = 'D:/trae/paper/input/features/test/featureselect/'
output_dir = 'D:/trae/paper/input/features/test/featureselect/output/'
features_df = pd.read_csv(f'{input_dir}features_processed.csv')
labels_df = pd.read_csv(f'{input_dir}labels.csv')
feature_names = pd.read_csv(f'{input_dir}feature_names.txt', header=None)

feature_names_list = feature_names[0].tolist()
print(f"    样本数量: {features_df.shape[0]}")
print(f"    原始特征数量: {features_df.shape[1] - 1}")

# 2. 数据预处理
print("\n[2] 数据预处理...")
X = features_df.iloc[:, 1:].values
y = labels_df['Label'].values
feature_names_array = np.array(feature_names_list)

print(f"    特征矩阵形状: {X.shape}")
print(f"    标签分布: 类别0={np.sum(y==0)}, 类别1={np.sum(y==1)}")

# 3. RFE特征选择 (使用固定特征数)
print("\n[3] 递归消除特征选择 (RFE)...")
print("    这可能需要几分钟时间...")

# 基模型
base_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=1
)

# 目标选择特征数
n_features_to_select = 100

# RFE
rfe = RFE(
    estimator=base_model,
    n_features_to_select=n_features_to_select,
    step=20  # 每次消除20个特征
)

rfe.fit(X, y)

# 4. 获取结果
print("\n[4] RFE结果分析...")
selected_mask = rfe.support_
ranking = rfe.ranking_
selected_features = feature_names_array[selected_mask]

# 按排名排序 (排名1为最重要)
ranking_selected = ranking[selected_mask]
sorted_idx = np.argsort(ranking_selected)
selected_features_sorted = selected_features[sorted_idx]

print(f"    选择的特征数量: {np.sum(selected_mask)} / {len(feature_names_list)}")

# 排名前20的特征
print(f"\n    排名前20的特征:")
for i in range(min(20, len(selected_features_sorted))):
    rank = ranking[np.where(feature_names_array == selected_features_sorted[i])[0][0]]
    print(f"        {i+1}. {selected_features_sorted[i]} (排名: {rank})")

# 5. 保存选择的特征
print("\n[5] 保存结果...")

selected_features_df = pd.DataFrame({
    'Feature_Name': selected_features_sorted,
    'RFE_Rank': ranking[selected_mask][sorted_idx]
})
selected_features_df.to_csv(f'{output_dir}rfe_selected_features.csv', index=False)
print(f"    已保存选择的特征到: rfe_selected_features.csv")

# 保存完整的排名结果
full_results_df = pd.DataFrame({
    'Feature_Name': feature_names_array,
    'RFE_Rank': ranking,
    'Selected': selected_mask
})
full_results_df.to_csv(f'{output_dir}rfe_full_results.csv', index=False)
print(f"    已保存完整结果到: rfe_full_results.csv")

# 6. 模型性能评估
print("\n[6] 模型性能评估...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 全特征性能
cv_scores_full = cross_val_score(base_model, X, y, cv=cv, scoring='accuracy')
print(f"    全特征5折交叉验证准确率: {cv_scores_full.mean():.4f} (+/- {cv_scores_full.std()*2:.4f})")

# 选中特征性能
X_selected = X[:, selected_mask]
cv_scores_selected = cross_val_score(base_model, X_selected, y, cv=cv, scoring='accuracy')
print(f"    选中特征5折交叉验证准确率: {cv_scores_selected.mean():.4f} (+/- {cv_scores_selected.std()*2:.4f})")

# 7. 可视化
print("\n[7] 生成可视化图表...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 7.1 排名分布直方图
ax1 = axes[0, 0]
ax1.hist(ranking, bins=30, edgecolor='black', alpha=0.7, color='darkorange')
ax1.axvline(x=1, color='red', linestyle='--', linewidth=1.5, label='Selected (rank=1)')
ax1.set_xlabel('Feature Ranking', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Distribution of Feature Rankings', fontsize=12, fontweight='bold')
ax1.legend()

# 7.2 选中特征的排名详情
ax2 = axes[0, 1]
selected_ranks = ranking[selected_mask]
top_n = min(30, len(selected_features_sorted))
ax2.barh(range(top_n), selected_ranks[:top_n], color='coral', edgecolor='black', alpha=0.8)
ax2.set_yticks(range(top_n))
ax2.set_yticklabels([name[:35] + '...' if len(name) > 35 else name for name in selected_features_sorted[:top_n]], fontsize=8)
ax2.set_xlabel('Feature Ranking (lower is better)', fontsize=11)
ax2.set_title('Top 30 Selected Features (by Ranking)', fontsize=12, fontweight='bold')
ax2.invert_yaxis()

# 7.3 不同特征数量的性能曲线
ax3 = axes[1, 0]
n_features_list = [50, 100, 150, 200, 300, 500]
cv_scores_list = []
for n_feat in n_features_list:
    rfe_temp = RFE(
        estimator=RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1),
        n_features_to_select=n_feat,
        step=20
    )
    rfe_temp.fit(X, y)
    X_temp = X[:, rfe_temp.support_]
    scores = cross_val_score(RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1), 
                            X_temp, y, cv=cv, scoring='accuracy')
    cv_scores_list.append(scores.mean())

ax3.plot(n_features_list, cv_scores_list, 'bo-', linewidth=2, markersize=8)
ax3.set_xlabel('Number of Features', fontsize=11)
ax3.set_ylabel('Cross-Validation Accuracy', fontsize=11)
ax3.set_title('Performance vs. Number of Features', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 7.4 特征选择前后对比
ax4 = axes[1, 1]
categories = ['Original\nFeatures', 'Selected\nFeatures']
values = [len(feature_names_list), np.sum(selected_mask)]
colors_bar = ['lightcoral', 'seagreen']
bars = ax4.bar(categories, values, color=colors_bar, edgecolor='black', alpha=0.8)
ax4.set_ylabel('Number of Features', fontsize=11)
ax4.set_title(f'Feature Selection Summary\n(Reduction: {(1-np.sum(selected_mask)/len(feature_names_list))*100:.1f}%)', 
              fontsize=12, fontweight='bold')
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
             str(val), ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}rfe_feature_selection.png', dpi=150, bbox_inches='tight')
print(f"    已保存可视化图表到: rfe_feature_selection.png")

# 8. 打印总结
print("\n" + "=" * 60)
print("递归消除特征选择总结")
print("=" * 60)
print(f"    原始特征数量: {len(feature_names_list)}")
print(f"    选择的特征数量: {np.sum(selected_mask)}")
print(f"    特征减少比例: {(1-np.sum(selected_mask)/len(feature_names_list))*100:.1f}%")
print(f"    全特征准确率: {cv_scores_full.mean():.4f} (+/- {cv_scores_full.std()*2:.4f})")
print(f"    选中特征准确率: {cv_scores_selected.mean():.4f} (+/- {cv_scores_selected.std()*2:.4f})")
print("\n结果文件已保存到:")
print(f"    - {output_dir}rfe_selected_features.csv")
print(f"    - {output_dir}rfe_full_results.csv")
print(f"    - {output_dir}rfe_feature_selection.png")
print("=" * 60)
