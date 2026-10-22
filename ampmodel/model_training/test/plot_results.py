"""
特征数比较可视化脚本
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = r"D:\trae\paper\input\features\test\featureselect\smalltest"
PLOT_DIR = r"D:\trae\paper\input\features\test\featureselect\smalltest\plots"

import os
os.makedirs(PLOT_DIR, exist_ok=True)

def load_results():
    """加载结果"""
    df = pd.read_csv(f"{OUTPUT_DIR}/feature_comparison.csv")
    return df

def plot_cv_f1_by_features(df):
    """绘制不同特征数下各模型的CV F1分数"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = df['Model'].unique()
    features = [50, 100, 200]
    
    x = np.arange(len(features))
    width = 0.12
    
    for i, model in enumerate(models):
        cv_f1_means = []
        cv_f1_stds = []
        for n_feat in features:
            row = df[(df['Model'] == model) & (df['N_Features'] == n_feat)]
            if not row.empty:
                # 从CV F1列提取均值
                f1_str = row['CV F1'].values[0]
                mean_val = float(f1_str.split(' (+/- ')[0])
                std_val = float(f1_str.split('(+/- ')[1].replace(')', ''))
                cv_f1_means.append(mean_val)
                cv_f1_stds.append(std_val)
            else:
                cv_f1_means.append(0)
                cv_f1_stds.append(0)
        
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.bar(x + offset, cv_f1_means, width, label=model, yerr=cv_f1_stds, capsize=3)
    
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('CV F1-Score', fontsize=12)
    ax.set_title('CV F1-Score vs Number of Features', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0.7, 0.95)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/cv_f1_by_features.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/cv_f1_by_features.png")
    plt.close()

def plot_test_f1_by_features(df):
    """绘制不同特征数下各模型的测试集F1分数"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = df['Model'].unique()
    features = [50, 100, 200]
    
    x = np.arange(len(features))
    width = 0.12
    
    for i, model in enumerate(models):
        test_f1_vals = []
        for n_feat in features:
            row = df[(df['Model'] == model) & (df['N_Features'] == n_feat)]
            if not row.empty:
                test_f1_vals.append(float(row['Test F1'].values[0]))
            else:
                test_f1_vals.append(0)
        
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.bar(x + offset, test_f1_vals, width, label=model)
    
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Test F1-Score', fontsize=12)
    ax.set_title('Test F1-Score vs Number of Features', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/test_f1_by_features.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/test_f1_by_features.png")
    plt.close()

def plot_best_config_heatmap(df):
    """绘制最佳配置热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = df['Model'].unique()
    features = [50, 100, 200]
    
    # CV F1 热力图
    cv_matrix = np.zeros((len(models), len(features)))
    test_matrix = np.zeros((len(models), len(features)))
    
    for i, model in enumerate(models):
        for j, n_feat in enumerate(features):
            row = df[(df['Model'] == model) & (df['N_Features'] == n_feat)]
            if not row.empty:
                f1_str = row['CV F1'].values[0]
                cv_matrix[i, j] = float(f1_str.split(' (+/- ')[0])
                test_matrix[i, j] = float(row['Test F1'].values[0])
    
    im1 = axes[0].imshow(cv_matrix, cmap='YlGn', aspect='auto', vmin=0.75, vmax=0.9)
    axes[0].set_xticks(range(len(features)))
    axes[0].set_xticklabels(features)
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels(models)
    axes[0].set_xlabel('Number of Features')
    axes[0].set_title('CV F1-Score')
    
    for i in range(len(models)):
        for j in range(len(features)):
            axes[0].text(j, i, f'{cv_matrix[i, j]:.3f}', ha='center', va='center', fontsize=9)
    
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(test_matrix, cmap='YlGn', aspect='auto', vmin=0.5, vmax=1.0)
    axes[1].set_xticks(range(len(features)))
    axes[1].set_xticklabels(features)
    axes[1].set_yticks(range(len(models)))
    axes[1].set_yticklabels(models)
    axes[1].set_xlabel('Number of Features')
    axes[1].set_title('Test F1-Score')
    
    for i in range(len(models)):
        for j in range(len(features)):
            axes[1].text(j, i, f'{test_matrix[i, j]:.3f}', ha='center', va='center', fontsize=9)
    
    plt.colorbar(im2, ax=axes[1])
    
    plt.suptitle('Model Performance Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/performance_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/performance_heatmap.png")
    plt.close()

def plot_model_comparison(df):
    """绘制模型比较图（按最佳特征数）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 获取每个模型最佳配置
    best_configs = []
    for model in df['Model'].unique():
        df_model = df[df['Model'] == model]
        best_row = df_model.loc[df_model['cv_f1_mean'].idxmax()]
        best_configs.append({
            'Model': model,
            'N_Features': int(best_row['N_Features']),
            'CV F1': best_row['cv_f1_mean'],
            'Test F1': float(best_row['Test F1'])
        })
    
    df_best = pd.DataFrame(best_configs)
    df_best = df_best.sort_values('CV F1', ascending=True)
    
    y_pos = np.arange(len(df_best))
    
    # CV F1
    colors = ['#2ecc71' if nf == 100 else '#3498db' for nf in df_best['N_Features']]
    axes[0].barh(y_pos, df_best['CV F1'], color=colors)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([f"{m} ({n}f)" for m, n in zip(df_best['Model'], df_best['N_Features'])])
    axes[0].set_xlabel('CV F1-Score')
    axes[0].set_title('Best CV F1 by Model\n(100=green, others=blue)')
    axes[0].set_xlim(0.75, 0.9)
    
    for i, v in enumerate(df_best['CV F1']):
        axes[0].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)
    
    # Test F1
    axes[1].barh(y_pos, df_best['Test F1'], color=colors)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f"{m} ({n}f)" for m, n in zip(df_best['Model'], df_best['N_Features'])])
    axes[1].set_xlabel('Test F1-Score')
    axes[1].set_title('Test F1 by Best Config')
    axes[1].set_xlim(0.5, 1.0)
    
    for i, v in enumerate(df_best['Test F1']):
        axes[1].text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/model_best_config.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/model_best_config.png")
    plt.close()

def main():
    print("Generating visualizations...")
    
    df = load_results()
    
    plot_cv_f1_by_features(df)
    plot_test_f1_by_features(df)
    plot_best_config_heatmap(df)
    plot_model_comparison(df)
    
    print("\nAll visualizations saved!")

if __name__ == "__main__":
    main()
