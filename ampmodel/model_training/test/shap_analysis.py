"""
SHAP特征可解释性分析脚本
功能：对最佳模型进行SHAP分析，生成多种可视化图表
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os
import warnings
warnings.filterwarnings('ignore')

# 配置参数
DATA_DIR = r"D:\trae\paper\input\features\test\featureselect\smalltest\rf_k100"
OUTPUT_DIR = r"D:\trae\paper\input\features\test\featureselect\smalltest\plots"
PLOT_DIR = os.path.join(OUTPUT_DIR, "shap")
RANDOM_STATE = 42

os.makedirs(PLOT_DIR, exist_ok=True)

def load_data():
    """加载数据"""
    print("=" * 60)
    print("Step 1: Load Data")
    print("=" * 60)
    
    X_all = np.load(os.path.join(DATA_DIR, "X_all.npy"))
    y_all = np.load(os.path.join(DATA_DIR, "y_all.npy"))
    feature_names = np.load(os.path.join(DATA_DIR, "feature_names.npy"), allow_pickle=True)
    test_idx = np.load(os.path.join(DATA_DIR, "test_idx.npy"))
    
    print(f"X_all shape: {X_all.shape}")
    print(f"y_all shape: {y_all.shape}")
    print(f"Feature names: {len(feature_names)}")
    
    return X_all, y_all, feature_names, test_idx

def train_best_model(X_train, y_train):
    """训练最佳模型（Random Forest）"""
    print("\n" + "=" * 60)
    print("Step 2: Train Model")
    print("=" * 60)
    
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Model trained: Random Forest")
    return model

def compute_shap_values(model, X_test, feature_names):
    """计算SHAP值"""
    print("\n" + "=" * 60)
    print("Step 3: Compute SHAP Values")
    print("=" * 60)
    
    # 使用TreeExplainer（适用于Random Forest等树模型）
    explainer = shap.TreeExplainer(model)
    
    # 计算SHAP值
    shap_values = explainer(X_test)
    
    # 设置特征名称
    shap_values.feature_names = list(feature_names)
    
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Base value shape: {shap_values.base_values.shape}")
    print(f"Base value (first sample): {shap_values.base_values[0, 1]:.4f}")
    
    # 对于二分类问题，取正类的SHAP值
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
    
    print(f"SHAP values shape (after processing): {shap_values.shape}")
    
    return explainer, shap_values

def plot_shap_summary(shap_values, plot_type='beeswarm'):
    """绘制SHAP汇总图"""
    print(f"\nGenerating {plot_type} plot...")
    
    plt.figure(figsize=(10, 8))
    
    if plot_type == 'beeswarm':
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.title('SHAP Beeswarm Plot - Feature Impact on Predictions', fontsize=12)
    elif plot_type == 'bar':
        shap.plots.bar(shap_values, max_display=20, show=False)
        plt.title('SHAP Bar Plot - Mean Absolute SHAP Values', fontsize=12)
    elif plot_type == 'violin':
        shap.plots.violin(shap_values, max_display=20, show=False)
        plt.title('SHAP Violin Plot - Feature Value Distribution', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/shap_{plot_type}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/shap_{plot_type}.png")
    plt.close()

def plot_shap_waterfall(shap_values, idx=0):
    """绘制单个样本的瀑布图"""
    print(f"\nGenerating waterfall plot for sample {idx}...")
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], max_display=20, show=False)
    plt.title(f'SHAP Waterfall - Sample {idx} Prediction Breakdown', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/shap_waterfall_sample_{idx}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/shap_waterfall_sample_{idx}.png")
    plt.close()

def plot_shap_scatter(shap_values, top_n=5):
    """绘制Top特征与SHAP值的关系图"""
    print(f"\nGenerating scatter plots for top {top_n} features...")
    
    # 计算平均绝对SHAP值
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
    
    for i, idx in enumerate(top_indices):
        plt.figure(figsize=(8, 5))
        feature_name = shap_values.feature_names[idx] if shap_values.feature_names else f"Feature {idx}"
        shap.plots.scatter(shap_values[:, idx], show=False, alpha=0.6)
        plt.title(f'SHAP Scatter - {feature_name}', fontsize=12)
        plt.xlabel(f'Feature Value ({feature_name})')
        plt.ylabel('SHAP Value')
        
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/shap_scatter_top{i+1}_feature_{idx}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOT_DIR}/shap_scatter_top{i+1}_feature_{idx}.png")
        plt.close()

def plot_shap_heatmap(shap_values, max_samples=50):
    """绘制SHAP热力图"""
    print(f"\nGenerating heatmap plot...")
    
    plt.figure(figsize=(12, 8))
    
    # 取前max_samples个样本
    shap_subset = shap_values[:max_samples]
    
    shap.plots.heatmap(shap_subset, max_display=20, show=False)
    plt.title('SHAP Heatmap - Feature Contributions Across Samples', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/shap_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/shap_heatmap.png")
    plt.close()

def plot_feature_importance_comparison(shap_values, model, feature_names):
    """比较SHAP特征重要性和模型内置特征重要性"""
    print("\n" + "=" * 60)
    print("Step 4: Feature Importance Comparison")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # SHAP特征重要性
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-15:][::-1]
    
    ax1 = axes[0]
    y_pos = np.arange(len(top_indices))
    ax1.barh(y_pos, mean_abs_shap[top_indices][::-1])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"F{idx}" for idx in top_indices[::-1]])
    ax1.set_xlabel('Mean |SHAP Value|')
    ax1.set_title('SHAP Feature Importance')
    
    # 模型内置特征重要性
    model_importance = model.feature_importances_
    model_top_indices = np.argsort(model_importance)[-15:][::-1]
    
    ax2 = axes[1]
    y_pos = np.arange(len(model_top_indices))
    ax2.barh(y_pos, model_importance[model_top_indices][::-1])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"F{idx}" for idx in model_top_indices[::-1]])
    ax2.set_xlabel('Feature Importance')
    ax2.set_title('Random Forest Feature Importance')
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/shap_importance_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/shap_importance_comparison.png")
    plt.close()
    
    # 保存特征重要性到CSV
    importance_df = pd.DataFrame({
        'Feature_Index': range(len(mean_abs_shap)),
        'SHAP_Importance': mean_abs_shap,
        'RF_Importance': model_importance,
        'Rank_Change': np.argsort(np.argsort(-mean_abs_shap)) - np.argsort(np.argsort(-model_importance))
    })
    importance_df = importance_df.sort_values('SHAP_Importance', ascending=False)
    importance_df.to_csv(f'{PLOT_DIR}/feature_importance.csv', index=False)
    print(f"Saved: {PLOT_DIR}/feature_importance.csv")
    
    return importance_df

def analyze_predictions(shap_values, y_true, y_pred):
    """分析预测结果"""
    print("\n" + "=" * 60)
    print("Step 5: Prediction Analysis")
    print("=" * 60)
    
    # 找出正确和错误预测
    correct = y_pred == y_true
    incorrect = ~correct
    
    print(f"Correct predictions: {np.sum(correct)}/{len(y_true)}")
    print(f"Incorrect predictions: {np.sum(incorrect)}/{len(y_true)}")
    
    # 比较SHAP分布
    if np.sum(incorrect) > 0:
        correct_shap = np.abs(shap_values.values[correct]).mean(axis=0)
        incorrect_shap = np.abs(shap_values.values[incorrect]).mean(axis=0)
        
        print(f"\nMean |SHAP| for correct: {np.mean(correct_shap):.4f}")
        print(f"Mean |SHAP| for incorrect: {np.mean(incorrect_shap):.4f}")
    
    return correct, incorrect

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("SHAP Feature Interpretability Analysis")
    print("=" * 60)
    
    # 加载数据
    X_all, y_all, feature_names, test_idx = load_data()
    
    # 训练模型（使用全部train+val数据）
    train_val_idx = np.concatenate([np.load(os.path.join(DATA_DIR, f"cv_{repeat}_{fold}_train_idx.npy")) 
                                     for repeat in range(1, 4) 
                                     for fold in range(1, 6)])
    train_val_idx = np.unique(train_val_idx)
    
    X_train = X_all[train_val_idx]
    y_train = y_all[train_val_idx]
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]
    
    model = train_best_model(X_train, y_train)
    
    # 计算SHAP值
    explainer, shap_values = compute_shap_values(model, X_test, feature_names)
    
    # 生成可视化
    print("\n" + "=" * 60)
    print("Step 4: Generate Visualizations")
    print("=" * 60)
    
    # 汇总图
    plot_shap_summary(shap_values, 'beeswarm')
    plot_shap_summary(shap_values, 'bar')
    plot_shap_summary(shap_values, 'violin')
    
    # 单样本瀑布图
    plot_shap_waterfall(shap_values, idx=0)
    plot_shap_waterfall(shap_values, idx=1)
    
    # 散点图
    plot_shap_scatter(shap_values, top_n=5)
    
    # 热力图
    plot_shap_heatmap(shap_values)
    
    # 特征重要性比较
    importance_df = plot_feature_importance_comparison(shap_values, model, feature_names)
    
    # 预测分析
    y_pred = model.predict(X_test)
    analyze_predictions(shap_values, y_test, y_pred)
    
    # 打印Top 10重要特征
    print("\n" + "=" * 60)
    print("Top 10 Important Features (by SHAP)")
    print("=" * 60)
    print(importance_df.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("SHAP Analysis Complete!")
    print(f"Plots saved to: {PLOT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
