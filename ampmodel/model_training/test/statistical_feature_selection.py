"""
统计检验特征选择脚本 (SelectKBest + ANOVA/Mutual Information)
用于从1098个特征中选择重要特征，为抗菌肽分类模型做准备
使用ANOVA F检验和互信息进行特征选择
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("统计检验特征选择分析")
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

# 3. ANOVA F检验特征选择
print("\n[3] 统计检验特征选择 (ANOVA F检验)...")

k_best = 100  # 选择前100个特征
selector = SelectKBest(score_func=f_classif, k=k_best)
selector.fit(X, y)

# 获取分数和选择的特征
scores = selector.scores_
pvalues = selector.pvalues_
selected_mask = selector.get_support()
selected_features = feature_names_array[selected_mask]
selected_scores = scores[selected_mask]
selected_pvalues = pvalues[selected_mask]

print(f"    选择特征数量: {k_best}")

# 按分数排序
sorted_idx = np.argsort(selected_scores)[::-1]
selected_features_sorted = selected_features[sorted_idx]
selected_scores_sorted = selected_scores[sorted_idx]
selected_pvalues_sorted = selected_pvalues[sorted_idx]

print(f"\n    Top 20 重要特征 (按F-score):")
for i in range(min(20, len(selected_features_sorted))):
    print(f"        {i+1}. {selected_features_sorted[i]}: F={selected_scores_sorted[i]:.2f}, p={selected_pvalues_sorted[i]:.2e}")

# 4. 额外分析 - 不同k值
print("\n[4] 不同k值的特征选择分析...")
k_values = [50, 100, 150, 200, 300]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in k_values:
    selector_k = SelectKBest(score_func=f_classif, k=k)
    X_k = selector_k.fit_transform(X, y)
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
    cv_scores = cross_val_score(rf, X_k, y, cv=cv, scoring='accuracy')
    print(f"    k={k}: 准确率 = {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# 5. 保存选择的特征
print("\n[5] 保存结果...")

selected_features_df = pd.DataFrame({
    'Feature_Name': selected_features_sorted,
    'F_Score': selected_scores_sorted,
    'P_Value': selected_pvalues_sorted
})
selected_features_df.to_csv(f'{output_dir}stat_selected_features.csv', index=False)
print(f"    已保存选择的特征到: stat_selected_features.csv")

# 保存完整的分数结果
full_results_df = pd.DataFrame({
    'Feature_Name': feature_names_array,
    'F_Score': scores,
    'P_Value': pvalues,
    'Selected': selected_mask
})
full_results_df.to_csv(f'{output_dir}stat_full_results.csv', index=False)
print(f"    已保存完整结果到: stat_full_results.csv")

# 6. 模型性能评估
print("\n[6] 模型性能评估...")
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=1)

# 全特征性能
cv_scores_full = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
print(f"    全特征5折交叉验证准确率: {cv_scores_full.mean():.4f} (+/- {cv_scores_full.std()*2:.4f})")

# 选中特征性能
X_selected = X[:, selected_mask]
cv_scores_selected = cross_val_score(rf, X_selected, y, cv=cv, scoring='accuracy')
print(f"    选中特征5折交叉验证准确率: {cv_scores_selected.mean():.4f} (+/- {cv_scores_selected.std()*2:.4f})")

# 7. 可视化
print("\n[7] 生成可视化图表...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 7.1 F分数分布直方图
ax1 = axes[0, 0]
# 只显示非零分数
valid_scores = scores[scores > 0]
ax1.hist(valid_scores, bins=50, edgecolor='black', alpha=0.7, color='purple')
ax1.axvline(x=selected_scores_sorted[-1], color='red', linestyle='--', linewidth=1.5, 
            label=f'Top {k_best} threshold')
ax1.set_xlabel('F-Score', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Distribution of ANOVA F-Scores', fontsize=12, fontweight='bold')
ax1.legend()

# 7.2 Top 30特征F分数条形图
ax2 = axes[0, 1]
top_n = min(30, len(selected_features_sorted))
ax2.barh(range(top_n), selected_scores_sorted[:top_n], color='mediumpurple', edgecolor='black', alpha=0.8)
ax2.set_yticks(range(top_n))
ax2.set_yticklabels([name[:35] + '...' if len(name) > 35 else name for name in selected_features_sorted[:top_n]], fontsize=8)
ax2.set_xlabel('F-Score', fontsize=11)
ax2.set_title('Top 30 Features (by F-Score)', fontsize=12, fontweight='bold')
ax2.invert_yaxis()

# 7.3 不同k值的性能曲线
ax3 = axes[1, 0]
k_values_plot = [20, 50, 100, 150, 200, 300, 500]
cv_scores_k = []
for k in k_values_plot:
    selector_k = SelectKBest(score_func=f_classif, k=k)
    X_k = selector_k.fit_transform(X, y)
    rf_temp = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
    scores_k = cross_val_score(rf_temp, X_k, y, cv=cv, scoring='accuracy')
    cv_scores_k.append(scores_k.mean())

ax3.plot(k_values_plot, cv_scores_k, 'go-', linewidth=2, markersize=8)
ax3.set_xlabel('Number of Features (k)', fontsize=11)
ax3.set_ylabel('Cross-Validation Accuracy', fontsize=11)
ax3.set_title('Performance vs. Number of Features', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 标记最佳k
best_k_idx = np.argmax(cv_scores_k)
ax3.axvline(x=k_values_plot[best_k_idx], color='red', linestyle='--', 
            label=f'Best k = {k_values_plot[best_k_idx]}')
ax3.legend()

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
plt.savefig(f'{output_dir}stat_feature_selection.png', dpi=150, bbox_inches='tight')
print(f"    已保存可视化图表到: stat_feature_selection.png")

# 8. 打印总结
print("\n" + "=" * 60)
print("统计检验特征选择总结")
print("=" * 60)
print(f"    原始特征数量: {len(feature_names_list)}")
print(f"    选择的特征数量: {np.sum(selected_mask)}")
print(f"    特征减少比例: {(1-np.sum(selected_mask)/len(feature_names_list))*100:.1f}%")
print(f"    全特征准确率: {cv_scores_full.mean():.4f} (+/- {cv_scores_full.std()*2:.4f})")
print(f"    选中特征准确率: {cv_scores_selected.mean():.4f} (+/- {cv_scores_selected.std()*2:.4f})")
print(f"    最佳k值: {k_values_plot[best_k_idx]} (准确率: {cv_scores_k[best_k_idx]:.4f})")
print("\n结果文件已保存到:")
print(f"    - {output_dir}stat_selected_features.csv")
print(f"    - {output_dir}stat_full_results.csv")
print(f"    - {output_dir}stat_feature_selection.png")
print("=" * 60)
