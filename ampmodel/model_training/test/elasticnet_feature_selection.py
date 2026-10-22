"""
ElasticNet特征选择脚本
用于从1098个特征中选择重要特征，为抗菌肽分类模型做准备
使用LogisticRegression + ElasticNet正则化进行分类
使用Pipeline和scikit-learn最佳实践
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, GridSearchCV, learning_curve
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
print("ElasticNet特征选择分析 (优化版)")
print("=" * 60)

# 1. 加载数据
print("\n[1] 加载数据...")
input_dir = 'D:/trae/paper/input/features/test/featureselect/'
output_dir = 'D:/trae/paper/input/features/test/featureselect/output/'
features_df = pd.read_csv(f'{input_dir}features_processed.csv')
labels_df = pd.read_csv(f'{input_dir}labels.csv')
feature_names = pd.read_csv(f'{input_dir}feature_names.txt', header=None)

# 获取特征名列表
feature_names_list = feature_names[0].tolist()
print(f"    样本数量: {features_df.shape[0]}")
print(f"    原始特征数量: {features_df.shape[1] - 1}")
print(f"    特征名称数量: {len(feature_names_list)}")

# 2. 数据预处理
print("\n[2] 数据预处理...")
X = features_df.iloc[:, 1:].values
y = labels_df['Label'].values
feature_names_array = np.array(feature_names_list)

print(f"    特征矩阵形状: {X.shape}")
print(f"    标签分布: 类别0={np.sum(y==0)}, 类别1={np.sum(y==1)}")

# 3. 使用Pipeline + GridSearchCV进行超参数搜索
print("\n[3] 使用Pipeline + GridSearchCV进行超参数搜索...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 创建Pipeline: 标准化 + 分类器
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        max_iter=5000,
        random_state=42,
        n_jobs=1
    ))
])

# 定义参数网格
param_grid = {
    'classifier__C': np.logspace(-3, 1, 15),
    'classifier__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
}

# 使用GridSearchCV进行超参数搜索
print("    执行网格搜索...")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=1,
    verbose=0,
    return_train_score=True
)
grid_search.fit(X, y)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"\n    最佳参数:")
print(f"        C (正则化强度倒数): {best_params['classifier__C']:.4f}")
print(f"        l1_ratio: {best_params['classifier__l1_ratio']:.2f}")
print(f"        最佳交叉验证准确率: {best_score:.4f}")

# 4. 使用最佳参数训练最终模型
print("\n[4] 训练最终模型...")
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=best_params['classifier__l1_ratio'],
        C=best_params['classifier__C'],
        max_iter=5000,
        random_state=42,
        n_jobs=1
    ))
])
final_pipeline.fit(X, y)

# 5. 获取非零系数（选择的特征）
print("\n[5] 特征选择结果...")
coefficients = final_pipeline.named_steps['classifier'].coef_[0]
selected_mask = coefficients != 0
selected_features_idx = np.where(selected_mask)[0]
selected_features = feature_names_array[selected_features_idx]
selected_coefficients = coefficients[selected_features_idx]

print(f"    选择的特征数量: {len(selected_features)} / {len(feature_names_list)}")
print(f"    特征选择比例: {len(selected_features)/len(feature_names_list)*100:.2f}%")

# 6. 按系数绝对值排序
sorted_idx = np.argsort(np.abs(selected_coefficients))[::-1]
selected_features_sorted = selected_features[sorted_idx]
selected_coefficients_sorted = selected_coefficients[sorted_idx]

print(f"\n    Top 20 重要特征:")
for i in range(min(20, len(selected_features_sorted))):
    sign = "+" if selected_coefficients_sorted[i] > 0 else "-"
    print(f"        {i+1}. {selected_features_sorted[i]}: {sign}{abs(selected_coefficients_sorted[i]):.4f}")

# 7. 使用多指标进行评估
print("\n[6] 使用多指标进行模型评估...")

scorers = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# 全特征评估
cv_results = cross_validate(final_pipeline, X, y, cv=cv, scoring=scorers, return_train_score=True)
print(f"\n    全特征交叉验证评估:")
print(f"        训练集准确率: {cv_results['train_accuracy'].mean():.4f} (+/- {cv_results['train_accuracy'].std()*2:.4f})")
print(f"        测试集准确率: {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std()*2:.4f})")
print(f"        测试集ROC-AUC: {cv_results['test_roc_auc'].mean():.4f} (+/- {cv_results['test_roc_auc'].std()*2:.4f})")
print(f"        测试集F1: {cv_results['test_f1'].mean():.4f} (+/- {cv_results['test_f1'].std()*2:.4f})")

# 8. 保存选择的特征
print("\n[7] 保存结果...")

# 保存选择的特征列表
selected_features_df = pd.DataFrame({
    'Feature_Name': selected_features_sorted,
    'Coefficient': selected_coefficients_sorted,
    'Abs_Coefficient': np.abs(selected_coefficients_sorted)
})
selected_features_df.to_csv(f'{output_dir}selected_features.csv', index=False)
print(f"    已保存选择的特征到: selected_features.csv")

# 保存完整的特征选择结果
full_results_df = pd.DataFrame({
    'Feature_Name': feature_names_array,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients),
    'Selected': selected_mask
})
full_results_df.to_csv(f'{output_dir}elasticnet_full_results.csv', index=False)
print(f"    已保存完整结果到: elasticnet_full_results.csv")

# 9. 可视化 (包含学习曲线)
print("\n[8] 生成可视化图表...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 9.1 系数分布直方图
ax1 = axes[0, 0]
ax1.hist(coefficients[selected_mask], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
ax1.set_xlabel('Coefficient Value', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Distribution of Selected Feature Coefficients', fontsize=12, fontweight='bold')

# 9.2 Top 30特征系数条形图
ax2 = axes[0, 1]
top_n = min(30, len(selected_features_sorted))
colors = ['forestgreen' if c > 0 else 'crimson' for c in selected_coefficients_sorted[:top_n]]
ax2.barh(range(top_n), selected_coefficients_sorted[:top_n], color=colors, edgecolor='black', alpha=0.8)
ax2.set_yticks(range(top_n))
ax2.set_yticklabels([name[:35] + '...' if len(name) > 35 else name for name in selected_features_sorted[:top_n]], fontsize=8)
ax2.set_xlabel('Coefficient Value', fontsize=11)
ax2.set_title('Top 30 Important Features', fontsize=12, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax2.invert_yaxis()

# 9.3 不同l1_ratio的性能对比
ax3 = axes[0, 2]
l1_scores = []
l1_ratios_plot = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
for l1 in l1_ratios_plot:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=l1,
            C=best_params['classifier__C'],
            max_iter=5000,
            random_state=42
        ))
    ])
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    l1_scores.append(scores.mean())

ax3.plot(l1_ratios_plot, l1_scores, 'bo-', linewidth=2, markersize=8)
ax3.axvline(x=best_params['classifier__l1_ratio'], color='red', linestyle='--', linewidth=1.5, 
            label=f'Best L1 ratio = {best_params["classifier__l1_ratio"]:.2f}')
ax3.set_xlabel('L1 Ratio', fontsize=11)
ax3.set_ylabel('Cross-Validation Accuracy', fontsize=11)
ax3.set_title('L1 Ratio vs. Cross-Validation Accuracy', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 9.4 特征选择前后对比
ax4 = axes[1, 0]
categories = ['Original\nFeatures', 'Selected\nFeatures']
values = [len(feature_names_list), len(selected_features)]
colors_bar = ['lightcoral', 'seagreen']
bars = ax4.bar(categories, values, color=colors_bar, edgecolor='black', alpha=0.8)
ax4.set_ylabel('Number of Features', fontsize=11)
ax4.set_title(f'Feature Selection Summary\n(Reduction: {(1-len(selected_features)/len(feature_names_list))*100:.1f}%)', 
              fontsize=12, fontweight='bold')
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
             str(val), ha='center', va='bottom', fontsize=14, fontweight='bold')

# 9.5 学习曲线
ax5 = axes[1, 1]
train_sizes, train_scores_lc, test_scores_lc = learning_curve(
    final_pipeline, X, y, 
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
ax5.set_title('Learning Curve', fontsize=12, fontweight='bold')
ax5.legend(loc='lower right')
ax5.grid(True, alpha=0.3)

# 9.6 多指标对比
ax6 = axes[1, 2]
metrics = ['Accuracy', 'ROC-AUC', 'F1', 'Precision', 'Recall']
scores_list = [cv_results['test_accuracy'].mean(), cv_results['test_roc_auc'].mean(), 
               cv_results['test_f1'].mean(), cv_results['test_precision'].mean(), 
               cv_results['test_recall'].mean()]
colors_metrics = ['steelblue', 'forestgreen', 'coral', 'gold', 'mediumpurple']
bars = ax6.bar(metrics, scores_list, color=colors_metrics, edgecolor='black', alpha=0.8)
ax6.set_ylabel('Score', fontsize=11)
ax6.set_title('Multi-Metric Evaluation', fontsize=12, fontweight='bold')
ax6.set_ylim(0.7, 1.0)
for bar, score in zip(bars, scores_list):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.3f}', 
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}elasticnet_feature_selection.png', dpi=150, bbox_inches='tight')
print(f"    已保存可视化图表到: elasticnet_feature_selection.png")

# 10. 打印总结
print("\n" + "=" * 60)
print("ElasticNet特征选择总结")
print("=" * 60)
print(f"    原始特征数量: {len(feature_names_list)}")
print(f"    选择的特征数量: {len(selected_features)}")
print(f"    特征减少比例: {(1-len(selected_features)/len(feature_names_list))*100:.1f}%")
print(f"    最佳C: {best_params['classifier__C']:.4f}")
print(f"    最佳l1_ratio: {best_params['classifier__l1_ratio']:.2f}")
print(f"\n    模型性能:")
print(f"        准确率: {cv_results['test_accuracy'].mean():.4f}")
print(f"        ROC-AUC: {cv_results['test_roc_auc'].mean():.4f}")
print(f"        F1: {cv_results['test_f1'].mean():.4f}")
print(f"        Precision: {cv_results['test_precision'].mean():.4f}")
print(f"        Recall: {cv_results['test_recall'].mean():.4f}")
print("\n结果文件已保存到:")
print(f"    - {output_dir}selected_features.csv")
print(f"    - {output_dir}elasticnet_full_results.csv")
print(f"    - {output_dir}elasticnet_feature_selection.png")
print("=" * 60)
