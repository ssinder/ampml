"""
生成顶刊级别可视化图表 (优化版)
功能：使用优化后的模型生成高质量图表
- ROC曲线
- 混淆矩阵
- 精确率-召回率曲线
- 模型对比图
优化内容：
1. 修复模型文件名路径匹配问题
2. 动态检测模型文件是否存在
3. 支持 Pipeline 模型格式
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# 设置路径
MODEL_DIR = "results/models"
TEST_PATH = "data/test_data_selected.csv"
OUTPUT_DIR = "results"
FIGURE_DIR = "figures"

os.makedirs(FIGURE_DIR, exist_ok=True)

# 颜色方案 - 专业学术配色
COLORS = {
    'Logistic Regression': '#2E86AB',
    'Random Forest': '#E94F37',
    'SVM': '#1B998B',
    'XGBoost': '#F39237',
    'Gradient Boosting': '#8E44AD',
    'Voting Ensemble': '#27AE60'
}

# 设置绘图风格 - 顶刊要求
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3
})


def load_models_and_data():
    """加载模型和数据 - 动态检测模型文件"""
    print("=" * 60)
    print("加载模型和数据")
    print("=" * 60)
    
    test_data = pd.read_csv(TEST_PATH)
    y_test = test_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    
    print(f"测试集: {X_test.shape}")
    
    # 动态检测可用的模型文件
    models = {}
    model_mapping = [
        ('logistic_regression_cv_model.pkl', 'Logistic Regression'),
        ('random_forest_cv_model.pkl', 'Random Forest'),
        ('svm_cv_model.pkl', 'SVM'),
        ('xgboost_cv_model.pkl', 'XGBoost'),
        ('gradient_boosting_cv_model.pkl', 'Gradient Boosting'),
        ('voting_ensemble_cv_model.pkl', 'Voting Ensemble')
    ]
    
    display_names = []
    available_models = []
    
    for model_file, display_name in model_mapping:
        model_path = f"{MODEL_DIR}/{model_file}"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            models[display_name] = model
            available_models.append(display_name)
            print(f"  ✓ 加载 {display_name}")
        else:
            print(f"  ⚠ 模型文件不存在: {model_file}")
    
    if not models:
        raise FileNotFoundError(f"未找到任何模型文件，请先运行模型训练脚本")
    
    print(f"\n成功加载 {len(models)} 个模型")
    
    # 不再单独加载scaler - Pipeline中已包含
    X_test_scaled = X_test  # 不需要单独标准化，Pipeline会处理
    
    return models, X_test, X_test_scaled, y_test, available_models


def get_model_input(model_name):
    """判断模型需要原始数据还是标准化数据"""
    # Pipeline 已内置标准化，根据模型类型决定输入
    return 'pipeline'  # 所有Pipeline模型使用相同接口


def plot_roc_curves(models, X_test, y_test, model_names, save_path):
    """图1: ROC曲线"""
    print("\n生成 ROC 曲线...")
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    for name in model_names:
        model = models[name]
        
        # Pipeline 模型直接预测
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=COLORS.get(name, '#333333'), lw=2.0,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='lower right', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 1: ROC curves saved to {save_path}")
    
    # 保存绘图数据
    roc_data = []
    for name in model_names:
        model = models[name]
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        for fp, tp, th in zip(fpr, tpr, thresholds):
            roc_data.append({
                'Model': name,
                'FPR': fp,
                'TPR': tp,
                'Threshold': th,
                'AUC': roc_auc
            })
    
    roc_df = pd.DataFrame(roc_data)
    roc_df.to_csv(f"{OUTPUT_DIR}/roc_curve_data.csv", index=False)
    print(f"✓ ROC数据已保存: {OUTPUT_DIR}/roc_curve_data.csv")


def plot_confusion_matrices(models, X_test, y_test, model_names, save_path):
    """图2: 混淆矩阵"""
    print("\n生成混淆矩阵...")
    
    n_models = len(model_names)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, name in enumerate(model_names):
        model = models[name]
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-AMP', 'AMP'])
        disp.plot(ax=axes[idx], cmap='Blues', values_format='d')
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
    
    # 隐藏多余的子图
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 2: Confusion matrices saved to {save_path}")
    
    # 保存混淆矩阵数据
    cm_data = []
    for name in model_names:
        model = models[name]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        if cm.shape == (2, 2):
            cm_data.append({
                'Model': name,
                'TN': int(cm[0, 0]),
                'FP': int(cm[0, 1]),
                'FN': int(cm[1, 0]),
                'TP': int(cm[1, 1])
            })
    
    cm_df = pd.DataFrame(cm_data)
    cm_df.to_csv(f"{OUTPUT_DIR}/confusion_matrix_data.csv", index=False)
    print(f"✓ 混淆矩阵数据已保存: {OUTPUT_DIR}/confusion_matrix_data.csv")


def plot_precision_recall_curves(models, X_test, y_test, model_names, save_path):
    """图3: 精确率-召回率曲线"""
    print("\n生成 Precision-Recall 曲线...")
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    for name in model_names:
        model = models[name]
        
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        ax.plot(recall, precision, color=COLORS.get(name, '#333333'), lw=2.0,
                label=f'{name} (AP = {avg_precision:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='lower left', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 3: Precision-Recall curves saved to {save_path}")
    
    # 保存PR曲线数据
    pr_data = []
    for name in model_names:
        model = models[name]
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        for p, r, th in zip(precision, recall, thresholds):
            pr_data.append({
                'Model': name,
                'Precision': p,
                'Recall': r,
                'Threshold': th,
                'Avg_Precision': avg_precision
            })
    
    pr_df = pd.DataFrame(pr_data)
    pr_df.to_csv(f"{OUTPUT_DIR}/precision_recall_curve_data.csv", index=False)
    print(f"✓ PR曲线数据已保存: {OUTPUT_DIR}/precision_recall_curve_data.csv")


def plot_model_comparison(models, X_test, y_test, model_names, save_path):
    """图4: 模型性能对比"""
    print("\n生成模型性能对比图...")
    
    metrics_data = []
    
    for name in model_names:
        model = models[name]
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics_data.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df)
    
    # 绘制分组柱状图
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(model_names))
    width = 0.15
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']
    
    for i, (metric, color) in enumerate(zip(metric_names, colors)):
        bars = ax.bar(x + i * width - 2*width, metrics_df[metric], width, 
                      label=metric, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    ax.set_ylim([0.7, 1.0])
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 4: Model comparison saved to {save_path}")
    
    # 保存评估指标数据
    metrics_df.to_csv(f"{OUTPUT_DIR}/model_comparison_data.csv", index=False)
    print(f"✓ 模型对比数据已保存: {OUTPUT_DIR}/model_comparison_data.csv")


def main():
    """主函数"""
    print("=" * 60)
    print("生成顶刊级别可视化图表 (优化版)")
    print("=" * 60)
    
    # 加载数据和模型
    models, X_test, X_test_scaled, y_test, model_names = load_models_and_data()
    
    # 生成图表
    plot_roc_curves(models, X_test, y_test, model_names, 
                   f"{FIGURE_DIR}/fig1_roc_curves.png")
    
    plot_confusion_matrices(models, X_test, y_test, model_names, 
                           f"{FIGURE_DIR}/fig2_confusion_matrices.png")
    
    plot_precision_recall_curves(models, X_test, y_test, model_names, 
                                f"{FIGURE_DIR}/fig3_precision_recall_curves.png")
    
    plot_model_comparison(models, X_test, y_test, model_names, 
                         f"{FIGURE_DIR}/fig4_model_comparison.png")
    
    print("\n" + "=" * 60)
    print("✓ 所有图表生成完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()