"""
顶刊级可视化脚本
功能：生成风格一致的高质量图表，适用于论文投稿
遵循Nature/Science等顶刊规范
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os

# 配置路径
OUTPUT_DIR = r"D:\trae\paper\input\features\test\featureselect\smalltest"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots", "publication")
os.makedirs(PLOT_DIR, exist_ok=True)

# Okabe-Ito 颜色调色板 (色盲友好)
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', 
             '#0072B2', '#D55E00', '#CC79A7', '#000000']

def setup_publication_style():
    """设置顶刊级样式"""
    # Nature期刊标准尺寸
    SINGLE_COLUMN = 3.5  # 89mm
    DOUBLE_COLUMN = 7.0   # 183mm
    HEIGHT = 2.5
    
    # 设置字体 - 使用Times New Roman (更通用，支持性好)
    # 确保使用Type 1兼容字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    
    # PDF字体嵌入设置
    plt.rcParams['pdf.fonttype'] = 42  # 转换为TrueType (Type 42)
    plt.rcParams['ps.fonttype'] = 42   # 转换为TrueType
    
    # 设置颜色
    mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=OKABE_ITO)
    
    # 坐标轴设置
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.linewidth'] = 1.0
    
    return SINGLE_COLUMN, DOUBLE_COLUMN, HEIGHT

def save_figure(fig, name, formats=['pdf', 'png'], dpi=300):
    """保存图表为顶刊格式"""
    for fmt in formats:
        if fmt == 'pdf':
            fig.savefig(f'{PLOT_DIR}/{name}.pdf', format='pdf', 
                       bbox_inches='tight', pad_inches=0.05)
        else:
            fig.savefig(f'{PLOT_DIR}/{name}.{fmt}', dpi=dpi, 
                       bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/{name}.*")
    plt.close(fig)

def load_model_comparison():
    """加载模型比较数据"""
    df = pd.read_csv(f'{OUTPUT_DIR}/feature_comparison.csv')
    return df

def load_feature_importance():
    """加载特征重要性数据"""
    df = pd.read_csv(f'{OUTPUT_DIR}/plots/shap/feature_importance.csv')
    return df

def plot_figure1_model_comparison():
    """图1: 模型性能比较 - 多特征数对比"""
    print("\n=== Creating Figure 1: Model Comparison ===")
    
    SINGLE, DOUBLE, HEIGHT = setup_publication_style()
    df = load_model_comparison()
    
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, HEIGHT+0.3))
    
    # 提取数值
    models = df['Model'].unique()
    features = [50, 100, 200]
    
    # 计算每个模型的平均值
    model_data = []
    for model in models:
        for feat in features:
            row = df[(df['Model'] == model) & (df['N_Features'] == feat)]
            if not row.empty:
                cv_f1 = row['cv_f1_mean'].values[0]
                test_f1_val = row['Test F1'].values[0]
                # 处理数值或字符串类型
                if isinstance(test_f1_val, str):
                    test_f1 = float(test_f1_val.replace('+', '').split(' ')[0])
                else:
                    test_f1 = float(test_f1_val)
                model_data.append({
                    'Model': model,
                    'Features': feat,
                    'CV F1': cv_f1,
                    'Test F1': test_f1
                })
    
    df_plot = pd.DataFrame(model_data)
    
    # 左图: CV F1
    ax1 = axes[0]
    pivot_cv = df_plot.pivot(index='Model', columns='Features', values='CV F1')
    pivot_cv = pivot_cv.reindex(pivot_cv.mean(axis=1).sort_values(ascending=False).index)
    
    pivot_cv.plot(kind='bar', ax=ax1, width=0.75, color=[OKABE_ITO[1], OKABE_ITO[0], OKABE_ITO[3]])
    ax1.set_xlabel('')
    ax1.set_ylabel('CV F1-Score', fontsize=9)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax1.legend(title='Features', frameon=False, loc='lower right')
    ax1.set_ylim(0.75, 0.90)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 右图: Test F1
    ax2 = axes[1]
    pivot_test = df_plot.pivot(index='Model', columns='Features', values='Test F1')
    pivot_test = pivot_test.reindex(pivot_cv.index)
    
    pivot_test.plot(kind='bar', ax=ax2, width=0.75, color=[OKABE_ITO[1], OKABE_ITO[0], OKABE_ITO[3]])
    ax2.set_xlabel('')
    ax2.set_ylabel('Test F1-Score', fontsize=9)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax2.legend(title='Features', frameon=False, loc='lower right')
    ax2.set_ylim(0.5, 1.0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, 'fig1_model_comparison')
    plt.close()

def plot_figure2_heatmap():
    """图2: 性能热力图"""
    print("\n=== Creating Figure 2: Performance Heatmap ===")
    
    SINGLE, DOUBLE, HEIGHT = setup_publication_style()
    df = load_model_comparison()
    
    fig, ax = plt.subplots(figsize=(SINGLE, HEIGHT+0.5))
    
    # 构建矩阵
    models = df['Model'].unique()
    features = [50, 100, 200]
    
    cv_matrix = np.zeros((len(models), len(features)))
    for i, model in enumerate(models):
        for j, feat in enumerate(features):
            row = df[(df['Model'] == model) & (df['N_Features'] == feat)]
            if not row.empty:
                cv_matrix[i, j] = row['cv_f1_mean'].values[0]
    
    # 排序
    order = np.argsort(cv_matrix.mean(axis=1))[::-1]
    cv_matrix = cv_matrix[order]
    models = [models[i] for i in order]
    
    # 绘制热力图
    im = ax.imshow(cv_matrix, cmap='YlGn', aspect='auto', vmin=0.78, vmax=0.88)
    
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=7)
    ax.set_xlabel('Number of Features', fontsize=9)
    ax.set_title('CV F1-Score Heatmap', fontsize=9, pad=10)
    
    # 添加数值标注
    for i in range(len(models)):
        for j in range(len(features)):
            text = ax.text(j, i, f'{cv_matrix[i, j]:.3f}',
                          ha='center', va='center', fontsize=7,
                          color='white' if cv_matrix[i, j] < 0.82 else 'black')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('CV F1-Score', fontsize=8)
    
    plt.tight_layout()
    save_figure(fig, 'fig2_heatmap')
    plt.close()

def plot_figure3_feature_importance():
    """图3: 特征重要性比较"""
    print("\n=== Creating Figure 3: Feature Importance ===")
    
    SINGLE, DOUBLE, HEIGHT = setup_publication_style()
    df = load_feature_importance()
    
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, HEIGHT+0.3))
    
    # Top 15 特征
    df_top = df.nlargest(15, 'SHAP_Importance')
    
    # SHAP重要性
    ax1 = axes[0]
    y_pos = np.arange(len(df_top))
    ax1.barh(y_pos, df_top['SHAP_Importance'].values[::-1], color=OKABE_ITO[0], height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"F{int(idx)}" for idx in df_top['Feature_Index'].values[::-1]], fontsize=7)
    ax1.set_xlabel('Mean |SHAP Value|', fontsize=9)
    ax1.set_title('SHAP Feature Importance', fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # RF重要性
    ax2 = axes[1]
    df_top_rf = df.nlargest(15, 'RF_Importance')
    y_pos = np.arange(len(df_top_rf))
    ax2.barh(y_pos, df_top_rf['RF_Importance'].values[::-1], color=OKABE_ITO[1], height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"F{int(idx)}" for idx in df_top_rf['Feature_Index'].values[::-1]], fontsize=7)
    ax2.set_xlabel('Feature Importance', fontsize=9)
    ax2.set_title('Random Forest Importance', fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, 'fig3_feature_importance')
    plt.close()

def plot_figure4_best_models():
    """图4: 最佳模型比较条形图"""
    print("\n=== Creating Figure 4: Best Models ===")
    
    SINGLE, DOUBLE, HEIGHT = setup_publication_style()
    df = load_model_comparison()
    
    # 获取每个模型最佳配置
    best_configs = []
    for model in df['Model'].unique():
        df_model = df[df['Model'] == model]
        best_row = df_model.loc[df_model['cv_f1_mean'].idxmax()]
        test_f1_val = best_row['Test F1']
        if isinstance(test_f1_val, str):
            test_f1 = float(test_f1_val.replace('+', '').split(' ')[0])
        else:
            test_f1 = float(test_f1_val)
        best_configs.append({
            'Model': model,
            'Features': int(best_row['N_Features']),
            'CV F1': best_row['cv_f1_mean'],
            'CV AUC': best_row['cv_auc_mean'],
            'Test F1': test_f1
        })
    
    df_best = pd.DataFrame(best_configs)
    df_best = df_best.sort_values('CV F1', ascending=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, HEIGHT+0.5))
    
    # CV F1
    ax1 = axes[0]
    colors = [OKABE_ITO[2] if f == 100 else OKABE_ITO[3] for f in df_best['Features']]
    ax1.barh(range(len(df_best)), df_best['CV F1'], color=colors, height=0.7)
    ax1.set_yticks(range(len(df_best)))
    ax1.set_yticklabels([f"{m} ({f})" for m, f in zip(df_best['Model'], df_best['Features'])], fontsize=6)
    ax1.set_xlabel('CV F1-Score', fontsize=9)
    ax1.set_xlim(0.75, 0.90)
    ax1.set_title('CV F1', fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # CV AUC
    ax2 = axes[1]
    ax2.barh(range(len(df_best)), df_best['CV AUC'], color=colors, height=0.7)
    ax2.set_yticks(range(len(df_best)))
    ax2.set_yticklabels([])
    ax2.set_xlabel('CV ROC-AUC', fontsize=9)
    ax2.set_xlim(0.85, 0.95)
    ax2.set_title('CV AUC', fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Test F1
    ax3 = axes[2]
    ax3.barh(range(len(df_best)), df_best['Test F1'], color=colors, height=0.7)
    ax3.set_yticks(range(len(df_best)))
    ax3.set_yticklabels([])
    ax3.set_xlabel('Test F1-Score', fontsize=9)
    ax3.set_xlim(0.5, 1.0)
    ax3.set_title('Test F1', fontsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=OKABE_ITO[2], label='100 features'),
                       Patch(facecolor=OKABE_ITO[3], label='Other')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), frameon=False)
    
    plt.tight_layout()
    save_figure(fig, 'fig4_best_models')
    plt.close()

def plot_figure5_cv_by_features():
    """图5: 不同特征数下各模型CV F1趋势"""
    print("\n=== Creating Figure 5: CV F1 by Features ===")
    
    SINGLE, DOUBLE, HEIGHT = setup_publication_style()
    df = load_model_comparison()
    
    fig, ax = plt.subplots(figsize=(SINGLE, HEIGHT))
    
    models = ['SVM (RBF)', 'Random Forest', 'MLP', 'Logistic Regression', 'KNN']
    features = [50, 100, 200]
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, model in enumerate(models):
        cv_f1_vals = []
        for feat in features:
            row = df[(df['Model'] == model) & (df['N_Features'] == feat)]
            if not row.empty:
                cv_f1_vals.append(row['cv_f1_mean'].values[0])
            else:
                cv_f1_vals.append(np.nan)
        
        ax.plot(features, cv_f1_vals, marker=markers[i], 
                label=model, color=OKABE_ITO[i], linewidth=1.5, markersize=5)
    
    ax.set_xlabel('Number of Features', fontsize=9)
    ax.set_ylabel('CV F1-Score', fontsize=9)
    ax.set_xticks(features)
    ax.set_ylim(0.75, 0.90)
    ax.legend(loc='lower left', frameon=False, fontsize=6, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_figure(fig, 'fig5_cv_trend')
    plt.close()

def plot_figure6_summary():
    """图6: 实验总结图"""
    print("\n=== Creating Figure 6: Summary ===")
    
    SINGLE, DOUBLE, HEIGHT = setup_publication_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, HEIGHT))
    
    # 配置信息
    configs = ['50', '100', '200']
    best_models = ['RF', 'SVM(RBF)', 'SVM(RBF)']
    cv_scores = [0.855, 0.859, 0.830]
    test_scores = [0.737, 0.800, 0.900]
    
    # 柱状图 - CV分数
    ax1 = axes[0]
    x = np.arange(len(configs))
    width = 0.35
    bars1 = ax1.bar(x - width/2, cv_scores, width, label='CV F1', color=OKABE_ITO[0])
    bars2 = ax1.bar(x + width/2, test_scores, width, label='Test F1', color=OKABE_ITO[2])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{c}\n({bm})' for c, bm in zip(configs, best_models)], fontsize=7)
    ax1.set_ylabel('F1-Score', fontsize=9)
    ax1.legend(frameon=False, fontsize=7)
    ax1.set_ylim(0.6, 1.0)
    ax1.set_title('Performance by Feature Count', fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 添加数值标注
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.3f}',
                ha='center', va='bottom', fontsize=6)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.3f}',
                ha='center', va='bottom', fontsize=6)
    
    # 饼图 - 最佳配置分布
    ax2 = axes[1]
    feature_counts = {'50': 2, '100': 3, '200': 1}
    ax2.pie(feature_counts.values(), labels=feature_counts.keys(), 
            colors=[OKABE_ITO[1], OKABE_ITO[0], OKABE_ITO[3]], 
            autopct='%1.0f%%', startangle=90, textprops={'fontsize': 8})
    ax2.set_title('Best Config Distribution', fontsize=9)
    
    # 关键指标
    ax3 = axes[2]
    ax3.axis('off')
    metrics_text = """
    Key Findings:
    
    • Best CV: SVM (RBF) + 100 features
      CV F1 = 0.859
    
    • Best Test: SVM (RBF) + 200 features
      Test F1 = 0.900
    
    • Top 5 Features: 94, 99, 98, 93, 90
    
    • Total evaluations: 90 (6 models × 3 configs × 5 folds)
    """
    ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes, 
             fontsize=8, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    save_figure(fig, 'fig6_summary')
    plt.close()

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Creating Publication-Quality Figures")
    print("=" * 60)
    
    setup_publication_style()
    
    # 生成所有图表
    plot_figure1_model_comparison()
    plot_figure2_heatmap()
    plot_figure3_feature_importance()
    plot_figure4_best_models()
    plot_figure5_cv_by_features()
    plot_figure6_summary()
    
    print("\n" + "=" * 60)
    print("All Publication Figures Created!")
    print(f"Output: {PLOT_DIR}")
    print("=" * 60)
    
    # 列出生成的文件
    files = os.listdir(PLOT_DIR)
    print("\nGenerated files:")
    for f in sorted(files):
        print(f"  - {f}")

if __name__ == "__main__":
    main()
