"""
随机森林特征选择脚本
用于从1098个特征中选择重要特征，为抗菌肽分类模型做准备
基于随机森林的特征重要性进行特征选择
使用Pipeline和scikit-learn最佳实践
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, learning_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("随机森林特征选择分析 (优化版)")
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

# 3. 使用Pipeline进行特征选择和模型训练
print("\n[3] 使用Pipeline进行特征选择...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 使用SelectFromModel进行特征选择
rf_selector = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=1
)

# 训练选择器
rf_selector.fit(X, y)
importances = rf_selector.feature_importances_

# 使用不同阈值进行分析
thresholds = [0.001, 0.0009, 0.0005, 0.0001, np.mean(importances)]
print(f"    不同阈值下的特征选择数量:")
for thresh in thresholds:
    n_selected = np.sum(importances >= thresh)
    print(f"        阈值 {thresh:.6f}: {n_selected} 个特征")

# 使用均值作为阈值 (默认)
threshold = np.mean(importances)
selected_mask = importances >= threshold
selected_features_idx = np.where(selected_mask)[0]
selected_features = feature_names_array[selected_mask]
selected_importances = importances[selected_mask]

print(f"\n    使用阈值 (均值 {threshold:.6f}):")
print(f"    选择的特征数量: {np.sum(selected_mask)} / {len(feature_names_list)}")

# 按重要性排序
sorted_idx = np.argsort(selected_importances)[::-1]
selected_features_sorted = selected_features[sorted_idx]
selected_importances_sorted = selected_importances[sorted_idx]

print(f"\n    Top 20 重要特征:")
for i in range(min(20, len(selected_features_sorted))):
    print(f"        {i+1}. {selected_features_sorted[i]}: {selected_importances_sorted[i]:.4f}")

# 4. 使用Pipeline + 多指标评估
print("\n[4] 使用Pipeline进行模型评估 (多指标)...")

# 创建Pipeline: 特征选择 + 分类器
rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=1
)

# 定义scorers
scorers = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# 全特征评估
print("\n    全特征交叉验证评估:")
pipeline_full = Pipeline([
    ('classifier', rf_classifier)
])
cv_results_full = cross_validate(pipeline_full, X, y, cv=cv, scoring=scorers, return_train_score=True)
print(f"        训练集准确率: {cv_results_full['train_accuracy'].mean():.4f} (+/- {cv_results_full['train_accuracy'].std()*2:.4f})")
print(f"        测试集准确率: {cv_results_full['test_accuracy'].mean():.4f} (+/- {cv_results_full['test_accuracy'].std()*2:.4f})")
print(f"        测试集ROC-AUC: {cv_results_full['test_roc_auc'].mean():.4f} (+/- {cv_results_full['test_roc_auc'].std()*2:.4f})")
print(f"        测试集F1: {cv_results_full['test_f1'].mean():.4f} (+/- {cv_results_full['test_f1'].std()*2:.4f})")

# 选中特征评估
X_selected = X[:, selected_features_idx]
pipeline_selected = Pipeline([
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=1
    ))
])
cv_results_selected = cross_validate(pipeline_selected, X_selected, y, cv=cv, scoring=scorers, return_train_score=True)
print("\n    选中特征交叉验证评估:")
print(f"        训练集准确率: {cv_results_selected['train_accuracy'].mean():.4f} (+/- {cv_results_selected['train_accuracy'].std()*2:.4f})")
print(f"        测试集准确率: {cv_results_selected['test_accuracy'].mean():.4f} (+/- {cv_results_selected['test_accuracy'].std()*2:.4f})")
print(f"        测试集ROC-AUC: {cv_results_selected['test_roc_auc'].mean():.4f} (+/- {cv_results_selected['test_roc_auc'].std()*2:.4f})")
print(f"        测试集F1: {cv_results_selected['test_f1'].mean():.4f} (+/- {cv_results_selected['test_f1'].std()*2:.4f})")

# 5. 保存选择的特征
print("\n[5] 保存结果...")

selected_features_df = pd.DataFrame({
    'Feature_Name': selected_features_sorted,
    'Importance': selected_importances_sorted
})
selected_features_df.to_csv(f'{output_dir}rf_selected_features.csv', index=False)
print(f"    已保存选择的特征到: rf_selected_features.csv")

# 保存完整的特征重要性结果
full_results_df = pd.DataFrame({
    'Feature_Name': feature_names_array,
    'Importance': importances,
    'Selected': selected_mask
})
full_results_df.to_csv(f'{output_dir}rf_full_results.csv', index=False)
print(f"    已保存完整结果到: rf_full_results.csv")

# 6. 可视化 (包含学习曲线)
print("\n[6] 生成可视化图表...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 6.1 特征重要性分布
ax1 = axes[0, 0]
ax1.hist(importances, bins=50, edgecolor='black', alpha=0.7, color='forestgreen')
ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold = {threshold:.4f}')
ax1.set_xlabel('Feature Importance', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Distribution of Feature Importances', fontsize=12, fontweight='bold')
ax1.legend()

# 6.2 Top 30特征重要性条形图
ax2 = axes[0, 1]
top_n = min(30, len(selected_features_sorted))
ax2.barh(range(top_n), selected_importances_sorted[:top_n], color='steelblue', edgecolor='black', alpha=0.8)
ax2.set_yticks(range(top_n))
ax2.set_yticklabels([name[:35] + '...' if len(name) > 35 else name for name in selected_features_sorted[:top_n]], fontsize=8)
ax2.set_xlabel('Feature Importance', fontsize=11)
ax2.set_title('Top 30 Important Features', fontsize=12, fontweight='bold')
ax2.invert_yaxis()

# 6.3 累积重要性
ax3 = axes[0, 2]
sorted_importances = np.sort(importances)[::-1]
cumulative = np.cumsum(sorted_importances) / np.sum(importances) * 100
ax3.plot(range(1, len(cumulative)+1), cumulative, 'b-', linewidth=2)
ax3.axhline(y=80, color='red', linestyle='--', label='80% importance')
ax3.axhline(y=90, color='orange', linestyle='--', label='90% importance')
ax3.set_xlabel('Number of Features', fontsize=11)
ax3.set_ylabel('Cumulative Importance (%)', fontsize=11)
ax3.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 6.4 特征选择前后对比
ax4 = axes[1, 0]
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

# 6.5 学习曲线 (选中特征)
ax5 = axes[1, 1]
train_sizes, train_scores_lc, test_scores_lc = learning_curve(
    pipeline_selected, X_selected, y, 
    cv=cv, n_jobs=1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy', random_state=42
)
train_scores_mean = np.mean(train_scores_lc, axis=1)
test_scores_mean = np.mean(test_scores_lc, axis=1)
ax5.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training Score')
ax5.plot(train_sizes, test_scores_mean, 'o-', color='orange', label='Validation Score')
ax5.fill_between(train_sizes, train_scores_mean - np.std(train_scores_lc), 
                 train_scores_mean + np.std(train_scores_lc), alpha=0.1, color='blue')
ax5.fill_between(train_sizes, test_scores_mean - np.std(test_scores_lc), 
                 test_scores_mean + np.std(test_scores_lc), alpha=0.1, color='orange')
ax5.set_xlabel('Training Set Size', fontsize=11)
ax5.set_ylabel('Accuracy', fontsize=11)
ax5.set_title('Learning Curve (Selected Features)', fontsize=12, fontweight='bold')
ax5.legend(loc='lower right')
ax5.grid(True, alpha=0.3)

# 6.6 模型性能对比
ax6 = axes[1, 2]
metrics = ['Accuracy', 'ROC-AUC', 'F1']
full_scores = [cv_results_full['test_accuracy'].mean(), cv_results_full['test_roc_auc'].mean(), cv_results_full['test_f1'].mean()]
selected_scores = [cv_results_selected['test_accuracy'].mean(), cv_results_selected['test_roc_auc'].mean(), cv_results_selected['test_f1'].mean()]
x = np.arange(len(metrics))
width = 0.35
bars1 = ax6.bar(x - width/2, full_scores, width, label='Full Features', color='lightcoral', edgecolor='black')
bars2 = ax6.bar(x + width/2, selected_scores, width, label='Selected Features', color='seagreen', edgecolor='black')
ax6.set_ylabel('Score', fontsize=11)
ax6.set_title('Performance Comparison', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics)
ax6.legend()
ax6.set_ylim(0.7, 1.0)
for bar, score in zip(bars1, full_scores):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.3f}', 
             ha='center', va='bottom', fontsize=9)
for bar, score in zip(bars2, selected_scores):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.3f}', 
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}rf_feature_selection.png', dpi=150, bbox_inches='tight')
print(f"    已保存可视化图表到: rf_feature_selection.png")

# 7. 打印总结
print("\n" + "=" * 60)
print("随机森林特征选择总结")
print("=" * 60)
print(f"    原始特征数量: {len(feature_names_list)}")
print(f"    选择的特征数量: {np.sum(selected_mask)}")
print(f"    特征减少比例: {(1-np.sum(selected_mask)/len(feature_names_list))*100:.1f}%")
print(f"\n    全特征性能:")
print(f"        准确率: {cv_results_full['test_accuracy'].mean():.4f}")
print(f"        ROC-AUC: {cv_results_full['test_roc_auc'].mean():.4f}")
print(f"        F1: {cv_results_full['test_f1'].mean():.4f}")
print(f"\n    选中特征性能:")
print(f"        准确率: {cv_results_selected['test_accuracy'].mean():.4f}")
print(f"        ROC-AUC: {cv_results_selected['test_roc_auc'].mean():.4f}")
print(f"        F1: {cv_results_selected['test_f1'].mean():.4f}")
print("\n结果文件已保存到:")
print(f"    - {output_dir}rf_selected_features.csv")
print(f"    - {output_dir}rf_full_results.csv")
print(f"    - {output_dir}rf_feature_selection.png")
print("=" * 60)
