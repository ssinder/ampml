#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
抗菌肽序列统计分析 - 顶刊级别可视化
生成 publication-ready 图表，避免 Type3 字体
遵循 Nature/Science 等期刊的图表规范
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from string import ascii_uppercase

# 使用虚拟环境中的 Python
PYTHON_BIN = 'd:/trae/paper/input/venv/Scripts/python.exe'

# 设置后端为 PDF (避免交互式显示问题)
mpl.use('Agg')

# ============================================================
# 1. 期刊风格配置 - 避免 Type3 字体，使用 Type1 (TrueType)
# ============================================================
# 使用 Type1/TrueType 字体，避免 Type3 字体
plt.rcParams['pdf.fonttype'] = 42  # 输出 TrueType 字体
plt.rcParams['ps.fonttype'] = 42   # 输出 TrueType 字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['axes.titlesize'] = 9

# 颜色盲友好调色板 - Okabe-Ito
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=OKABE_ITO)

# 设置 seaborn 风格
import seaborn as sns
sns.set_style('ticks')
sns.set_context('paper', font_scale=1.1)

# ============================================================
# 2. 数据加载与预处理
# ============================================================
DATA_PATH = 'd:/trae/paper/data/sample/id_sequence_extracted.csv'
OUTPUT_DIR = 'd:/trae/paper/data/sample/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("加载序列数据...")
df = pd.read_csv(DATA_PATH)
df['seq_length'] = df['sequence'].str.len()

# 氨基酸组成分析
all_sequences = ''.join(df['sequence'].tolist())
standard_aa = 'ACDEFGHIKLMNPQRSTVWY'
aa_counts = {aa: all_sequences.count(aa) for aa in standard_aa}
aa_freq = {aa: count / len(all_sequences) * 100 for aa, count in aa_counts.items()}

# 氨基酸性质分组
hydrophobic = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']
polar = ['S', 'T', 'N', 'Q', 'Y', 'C']
charged_pos = ['K', 'R', 'H']
charged_neg = ['D', 'E']

hydrophobic_freq = sum(aa_freq.get(aa, 0) for aa in hydrophobic)
polar_freq = sum(aa_freq.get(aa, 0) for aa in polar)
pos_freq = sum(aa_freq.get(aa, 0) for aa in charged_pos)
neg_freq = sum(aa_freq.get(aa, 0) for aa in charged_neg)

# ============================================================
# 3. 创建多面板顶刊级别图表
# ============================================================
def save_figure(fig, name, formats=['pdf', 'png'], dpi=300):
    """保存图表为多种格式"""
    for fmt in formats:
        path = os.path.join(OUTPUT_DIR, f'{name}.{fmt}')
        fig.savefig(path, dpi=dpi if fmt == 'png' else None, 
                    bbox_inches='tight', format=fmt)
        print(f"已保存: {path}")
    plt.close(fig)

# --- 图1: 序列长度分布 ---
print("生成图1: 序列长度分布...")
fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

# 直方图
ax1 = axes[0]
ax1.hist(df['seq_length'], bins=30, color=OKABE_ITO[0], edgecolor='white', alpha=0.8)
ax1.axvline(df['seq_length'].mean(), color=OKABE_ITO[5], linestyle='--', 
            linewidth=1.5, label=f'Mean = {df["seq_length"].mean():.1f}')
ax1.axvline(df['seq_length'].median(), color=OKABE_ITO[1], linestyle=':', 
            linewidth=1.5, label=f'Median = {df["seq_length"].median():.1f}')
ax1.set_xlabel('Sequence Length (amino acids)')
ax1.set_ylabel('Count')
ax1.legend(frameon=False)
sns.despine(ax=ax1)
ax1.text(-0.12, 1.05, 'A', transform=ax1.transAxes, fontsize=10, fontweight='bold')

# 箱线图
ax2 = axes[1]
bp = ax2.boxplot(df['seq_length'], vert=True, patch_artist=True,
                 widths=0.6, showfliers=True)
bp['boxes'][0].set_facecolor(OKABE_ITO[2])
bp['boxes'][0].set_alpha(0.7)
ax2.set_ylabel('Sequence Length (amino acids)')
ax2.set_xticklabels(['Antimicrobial\nPeptides'])
sns.despine(ax=ax2)
ax2.text(-0.12, 1.05, 'B', transform=ax2.transAxes, fontsize=10, fontweight='bold')

plt.tight_layout()
save_figure(fig, 'figure1_sequence_length_distribution')

# --- 图2: 氨基酸组成分析 ---
print("生成图2: 氨基酸组成分析...")
fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

# 氨基酸频率条形图
ax1 = axes[0]
sorted_aa = sorted(aa_freq.items(), key=lambda x: x[1], reverse=True)
aa_labels = [item[0] for item in sorted_aa]
aa_values = [item[1] for item in sorted_aa]
colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(aa_labels))]
bars = ax1.bar(aa_labels, aa_values, color=colors, edgecolor='white', linewidth=0.5)
ax1.set_xlabel('Amino Acid')
ax1.set_ylabel('Frequency (%)')
ax1.set_ylim(0, max(aa_values) * 1.15)
sns.despine(ax=ax1)
ax1.text(-0.08, 1.05, 'A', transform=ax1.transAxes, fontsize=10, fontweight='bold')

# 添加标注
for i, (aa, val) in enumerate(zip(aa_labels[:5], aa_values[:5])):
    ax1.annotate(f'{val:.1f}%', xy=(i, val), ha='center', va='bottom', 
                 fontsize=6, color='black')

# 氨基酸性质分组饼图
ax2 = axes[1]
groups = ['Hydrophobic\n(A,V,I,L,M,F,W,P)', 'Polar\n(S,T,N,Q,Y,C)', 
          'Positive\n(K,R,H)', 'Negative\n(D,E)']
group_freqs = [hydrophobic_freq, polar_freq, pos_freq, neg_freq]
colors_pie = [OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[1], OKABE_ITO[5]]
wedges, texts, autotexts = ax2.pie(group_freqs, labels=groups, autopct='%1.1f%%',
                                    colors=colors_pie, startangle=90,
                                    pctdistance=0.75, wedgeprops=dict(width=0.5))
for autotext in autotexts:
    autotext.set_fontsize(7)
    autotext.set_color('white')
    autotext.set_weight('bold')
ax2.text(-0.08, 1.05, 'B', transform=ax2.transAxes, fontsize=10, fontweight='bold')

plt.tight_layout()
save_figure(fig, 'figure2_amino_acid_composition')

# --- 图3: 长度分布分类统计 ---
print("生成图3: 长度分布分类统计...")
fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

# 长度分类统计
length_bins = [0, 10, 20, 30, 40, 50, 100]
length_labels = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-100']
df['length_bin'] = pd.cut(df['seq_length'], bins=length_bins, labels=length_labels)
length_counts = df['length_bin'].value_counts().sort_index()

ax1 = axes[0]
bars = ax1.bar(length_counts.index, length_counts.values, 
               color=OKABE_ITO[:len(length_counts)], edgecolor='white')
ax1.set_xlabel('Sequence Length Range')
ax1.set_ylabel('Number of Sequences')
ax1.set_ylim(0, max(length_counts.values) * 1.15)
sns.despine(ax=ax1)

# 添加数值标注
for i, (idx, val) in enumerate(length_counts.items()):
    ax1.annotate(f'{val}', xy=(i, val), ha='center', va='bottom', fontsize=8)
ax1.text(-0.08, 1.05, 'A', transform=ax1.transAxes, fontsize=10, fontweight='bold')

# 累积分布函数
ax2 = axes[1]
sorted_lengths = np.sort(df['seq_length'])
cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
ax2.plot(sorted_lengths, cumulative, color=OKABE_ITO[0], linewidth=1.5)
ax2.fill_between(sorted_lengths, 0, cumulative, alpha=0.3, color=OKABE_ITO[0])
ax2.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax2.axhline(0.9, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax2.set_xlabel('Sequence Length (amino acids)')
ax2.set_ylabel('Cumulative Proportion')
ax2.set_ylim(0, 1.05)
sns.despine(ax=ax2)

# 标注中位数和90%分位数
median_len = df['seq_length'].median()
p90_len = df['seq_length'].quantile(0.9)
ax2.axvline(median_len, color=OKABE_ITO[5], linestyle=':', linewidth=1.2)
ax2.axvline(p90_len, color=OKABE_ITO[1], linestyle=':', linewidth=1.2)
ax2.annotate(f'Median={median_len:.0f}', xy=(median_len, 0.5), 
             xytext=(median_len+5, 0.55), fontsize=7, color=OKABE_ITO[5])
ax2.annotate(f'P90={p90_len:.0f}', xy=(p90_len, 0.9), 
             xytext=(p90_len+5, 0.85), fontsize=7, color=OKABE_ITO[1])
ax2.text(-0.08, 1.05, 'B', transform=ax2.transAxes, fontsize=10, fontweight='bold')

plt.tight_layout()
save_figure(fig, 'figure3_length_distribution')

# --- 图4: 综合特征热力图 ---
print("生成图4: 氨基酸相关性热力图...")
fig, ax = plt.subplots(figsize=(5, 4))

# 创建氨基酸共现矩阵
aa_matrix = np.zeros((len(standard_aa), len(standard_aa)))
for seq in df['sequence']:
    seq_aa = [a for a in seq if a in standard_aa]
    for i, aa1 in enumerate(seq_aa):
        for j, aa2 in enumerate(seq_aa):
            if i != j:
                aa_matrix[standard_aa.index(aa1)][standard_aa.index(aa2)] += 1

# 归一化
row_sums = aa_matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
aa_matrix_norm = aa_matrix / row_sums

# 绘制热力图
im = ax.imshow(aa_matrix_norm, cmap='viridis', aspect='auto')
ax.set_xticks(range(len(standard_aa)))
ax.set_yticks(range(len(standard_aa)))
ax.set_xticklabels(list(standard_aa), fontsize=7)
ax.set_yticklabels(list(standard_aa), fontsize=7)
ax.set_xlabel('Amino Acid j')
ax.set_ylabel('Amino Acid i')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Conditional Probability', fontsize=8)

plt.tight_layout()
save_figure(fig, 'figure4_amino_acid_cooccurrence')

# --- 图5: 综合面板图 (用于论文) ---
print("生成图5: 综合面板图...")
fig = plt.figure(figsize=(8, 6))

# 使用 GridSpec 创建复杂布局
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35,
                       left=0.08, right=0.98, top=0.95, bottom=0.1)

# Panel A: 长度直方图
ax_a = fig.add_subplot(gs[0, 0])
ax_a.hist(df['seq_length'], bins=25, color=OKABE_ITO[0], edgecolor='white', alpha=0.8)
ax_a.axvline(df['seq_length'].mean(), color=OKABE_ITO[5], linestyle='--', linewidth=1.2)
ax_a.set_xlabel('Length (aa)')
ax_a.set_ylabel('Count')
sns.despine(ax=ax_a)
ax_a.text(-0.15, 1.12, 'A', transform=ax_a.transAxes, fontsize=11, fontweight='bold')

# Panel B: 氨基酸频率
ax_b = fig.add_subplot(gs[0, 1])
ax_b.bar(aa_labels, aa_values, color=colors, edgecolor='white', linewidth=0.3)
ax_b.set_xlabel('Amino Acid')
ax_b.set_ylabel('Frequency (%)')
ax_b.tick_params(axis='x', rotation=45, labelsize=6)
sns.despine(ax=ax_b)
ax_b.text(-0.15, 1.12, 'B', transform=ax_b.transAxes, fontsize=11, fontweight='bold')

# Panel C: 性质分组
ax_c = fig.add_subplot(gs[0, 2])
ax_c.barh(groups, group_freqs, color=colors_pie, edgecolor='white')
ax_c.set_xlabel('Frequency (%)')
ax_c.set_xlim(0, max(group_freqs) * 1.2)
sns.despine(ax=ax_c)
ax_c.text(-0.15, 1.12, 'C', transform=ax_c.transAxes, fontsize=11, fontweight='bold')

# Panel D: 长度分类
ax_d = fig.add_subplot(gs[1, 0])
bars = ax_d.bar(length_counts.index.astype(str), length_counts.values,
                color=OKABE_ITO[:len(length_counts)], edgecolor='white')
ax_d.set_xlabel('Length Range')
ax_d.set_ylabel('Count')
sns.despine(ax=ax_d)
ax_d.text(-0.15, 1.12, 'D', transform=ax_d.transAxes, fontsize=11, fontweight='bold')

# Panel E: 累积分布
ax_e = fig.add_subplot(gs[1, 1])
ax_e.plot(sorted_lengths, cumulative, color=OKABE_ITO[0], linewidth=1.5)
ax_e.fill_between(sorted_lengths, 0, cumulative, alpha=0.3, color=OKABE_ITO[0])
ax_e.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax_e.set_xlabel('Length (aa)')
ax_e.set_ylabel('Cumulative Proportion')
ax_e.set_ylim(0, 1.05)
sns.despine(ax=ax_e)
ax_e.text(-0.15, 1.12, 'E', transform=ax_e.transAxes, fontsize=11, fontweight='bold')

# Panel F: 统计摘要文字
ax_f = fig.add_subplot(gs[1, 2])
ax_f.axis('off')
summary_text = f"""Statistical Summary
====================
Total Sequences: {len(df)}
Mean Length: {df['seq_length'].mean():.1f} aa
Median Length: {df['seq_length'].median():.1f} aa
Std Deviation: {df['seq_length'].std():.1f} aa
Min Length: {df['seq_length'].min()} aa
Max Length: {df['seq_length'].max()} aa

Hydrophobic: {hydrophobic_freq:.1f}%
Polar: {polar_freq:.1f}%
Positive: {pos_freq:.1f}%
Negative: {neg_freq:.1f}%"""
ax_f.text(0.1, 0.95, summary_text, transform=ax_f.transAxes, 
          fontsize=8, verticalalignment='top', fontfamily='monospace')
ax_f.text(-0.15, 1.12, 'F', transform=ax_f.transAxes, fontsize=11, fontweight='bold')

save_figure(fig, 'figure5_comprehensive_summary')

print("\n" + "=" * 50)
print("所有图表生成完成!")
print(f"输出目录: {OUTPUT_DIR}")
print("=" * 50)

# 输出统计摘要
print(f"\n最终统计摘要:")
print(f"  总序列数: {len(df)}")
print(f"  平均长度: {df['seq_length'].mean():.2f} ± {df['seq_length'].std():.2f} aa")
print(f"  长度范围: {df['seq_length'].min()} - {df['seq_length'].max()} aa")
print(f"  疏水性氨基酸占比: {hydrophobic_freq:.1f}%")
print(f"  极性氨基酸占比: {polar_freq:.1f}%")
print(f"  正电荷氨基酸占比: {pos_freq:.1f}%")
print(f"  负电荷氨基酸占比: {neg_freq:.1f}%")
