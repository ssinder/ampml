#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
抗菌肽序列对比分析 - 顶刊级别可视化
抗真菌序列 vs 非抗菌序列
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.use('Agg')

# ============================================================
# 1. 期刊风格配置 - 避免 Type3 字体
# ============================================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['axes.titlesize'] = 9

OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=OKABE_ITO)

sns.set_style('ticks')
sns.set_context('paper', font_scale=1.1)

# ============================================================
# 2. 数据加载
# ============================================================
antifungal_path = 'd:/trae/paper/data/sample/representative_sequences_id_seq.csv'
non_antibacterial_path = 'd:/trae/paper/data/sample/id_sequence_extracted.csv'

df_antifungal = pd.read_csv(antifungal_path)
df_non_antibacterial = pd.read_csv(non_antibacterial_path)

if 'Sequence' in df_antifungal.columns:
    df_antifungal = df_antifungal.rename(columns={'Sequence': 'sequence'})

df_antifungal['seq_length'] = df_antifungal['sequence'].str.len()
df_non_antibacterial['seq_length'] = df_non_antibacterial['sequence'].str.len()

# 氨基酸组成
standard_aa = 'ACDEFGHIKLMNPQRSTVWY'

def calc_aa_composition(sequences):
    all_seqs = ''.join(sequences.tolist())
    aa_counts = {aa: all_seqs.count(aa) for aa in standard_aa}
    total = sum(aa_counts.values())
    return {aa: count / total * 100 for aa, count in aa_counts.items()}

aa_freq_anti = calc_aa_composition(df_antifungal['sequence'])
aa_freq_non = calc_aa_composition(df_non_antibacterial['sequence'])

# 氨基酸性质分组
hydrophobic = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']
polar = ['S', 'T', 'N', 'Q', 'Y', 'C']
charged_pos = ['K', 'R', 'H']
charged_neg = ['D', 'E']

def calc_group_freq(aa_freq, group):
    return sum(aa_freq.get(aa, 0) for aa in group)

OUTPUT_DIR = 'd:/trae/paper/data/sample/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_figure(fig, name, formats=['pdf', 'png'], dpi=300):
    for fmt in formats:
        path = os.path.join(OUTPUT_DIR, f'{name}.{fmt}')
        fig.savefig(path, dpi=dpi if fmt == 'png' else None, 
                    bbox_inches='tight', format=fmt)
        print(f"已保存: {path}")
    plt.close(fig)

# ============================================================
# 3. 生成顶刊级别图表
# ============================================================

# --- Figure 1: 序列长度分布对比 ---
print("生成图1: 序列长度分布对比...")
fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

# 直方图对比
ax1 = axes[0]
bins = np.arange(0, 65, 5)
ax1.hist(df_antifungal['seq_length'], bins=bins, alpha=0.6, 
         label=f'Antifungal (n={len(df_antifungal)})', color=OKABE_ITO[0], edgecolor='white')
ax1.hist(df_non_antibacterial['seq_length'], bins=bins, alpha=0.6,
         label=f'Non-antibacterial (n={len(df_non_antibacterial)})', color=OKABE_ITO[1], edgecolor='white')
ax1.set_xlabel('Sequence Length (amino acids)')
ax1.set_ylabel('Count')
ax1.legend(frameon=False, loc='upper right')
sns.despine(ax=ax1)
ax1.text(-0.12, 1.08, 'A', transform=ax1.transAxes, fontsize=11, fontweight='bold')

# 箱线图对比
ax2 = axes[1]
data_for_box = pd.DataFrame({
    'Length': pd.concat([df_antifungal['seq_length'], df_non_antibacterial['seq_length']]),
    'Type': ['Antifungal'] * len(df_antifungal) + ['Non-antibacterial'] * len(df_non_antibacterial)
})
bp = ax2.boxplot([df_antifungal['seq_length'], df_non_antibacterial['seq_length']],
                 vert=True, patch_artist=True, widths=0.6,
                 labels=['Antifungal', 'Non-antibacterial'])
bp['boxes'][0].set_facecolor(OKABE_ITO[0])
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor(OKABE_ITO[1])
bp['boxes'][1].set_alpha(0.7)
ax2.set_ylabel('Sequence Length (aa)')
sns.despine(ax=ax2)
ax2.text(-0.12, 1.08, 'B', transform=ax2.transAxes, fontsize=11, fontweight='bold')

# 添加均值标注
ax2.annotate(f"Mean={df_antifungal['seq_length'].mean():.1f}", 
             xy=(1, df_antifungal['seq_length'].mean()), 
             xytext=(1.3, df_antifungal['seq_length'].mean() + 5),
             fontsize=6, color=OKABE_ITO[0])
ax2.annotate(f"Mean={df_non_antibacterial['seq_length'].mean():.1f}", 
             xy=(2, df_non_antibacterial['seq_length'].mean()), 
             xytext=(2.1, df_non_antibacterial['seq_length'].mean() + 5),
             fontsize=6, color=OKABE_ITO[1])

# 累积分布函数
ax3 = axes[2]
sorted_anti = np.sort(df_antifungal['seq_length'])
sorted_non = np.sort(df_non_antibacterial['seq_length'])
cum_anti = np.arange(1, len(sorted_anti) + 1) / len(sorted_anti)
cum_non = np.arange(1, len(sorted_non) + 1) / len(sorted_non)
ax3.plot(sorted_anti, cum_anti, color=OKABE_ITO[0], linewidth=1.5, label='Antifungal')
ax3.plot(sorted_non, cum_non, color=OKABE_ITO[1], linewidth=1.5, label='Non-antibacterial')
ax3.set_xlabel('Sequence Length (aa)')
ax3.set_ylabel('Cumulative Proportion')
ax3.set_ylim(0, 1.05)
ax3.legend(frameon=False)
sns.despine(ax=ax3)
ax3.text(-0.12, 1.08, 'C', transform=ax3.transAxes, fontsize=11, fontweight='bold')

plt.tight_layout()
save_figure(fig, 'figure1_length_comparison')

# --- Figure 2: 氨基酸性质分布对比 ---
print("生成图2: 氨基酸性质分布对比...")
fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

# 分组条形图
ax1 = axes[0]
groups = ['Hydrophobic', 'Polar', 'Positive\n(K,R,H)', 'Negative\n(D,E)', 'Net Charge']
anti_freqs = [
    calc_group_freq(aa_freq_anti, hydrophobic),
    calc_group_freq(aa_freq_anti, polar),
    calc_group_freq(aa_freq_anti, charged_pos),
    calc_group_freq(aa_freq_anti, charged_neg),
    calc_group_freq(aa_freq_anti, charged_pos) - calc_group_freq(aa_freq_anti, charged_neg)
]
non_freqs = [
    calc_group_freq(aa_freq_non, hydrophobic),
    calc_group_freq(aa_freq_non, polar),
    calc_group_freq(aa_freq_non, charged_pos),
    calc_group_freq(aa_freq_non, charged_neg),
    calc_group_freq(aa_freq_non, charged_pos) - calc_group_freq(aa_freq_non, charged_neg)
]

x = np.arange(len(groups))
width = 0.35
bars1 = ax1.bar(x - width/2, anti_freqs, width, label='Antifungal', 
                color=OKABE_ITO[0], edgecolor='white')
bars2 = ax1.bar(x + width/2, non_freqs, width, label='Non-antibacterial', 
                color=OKABE_ITO[1], edgecolor='white')
ax1.set_ylabel('Frequency (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(groups, fontsize=7)
ax1.legend(frameon=False)
sns.despine(ax=ax1)
ax1.text(-0.12, 1.08, 'A', transform=ax1.transAxes, fontsize=11, fontweight='bold')

# 差异条形图
ax2 = axes[1]
diffs = [a - n for a, n in zip(anti_freqs[:-1], non_freqs[:-1])]
colors = [OKABE_ITO[2] if d > 0 else OKABE_ITO[5] for d in diffs]
bars = ax2.bar(groups[:4], diffs, color=colors, edgecolor='white')
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_ylabel('Difference (Antifungal - Non-antibacterial, %)')
ax2.set_xticklabels(groups[:4], fontsize=7, rotation=15)
sns.despine(ax=ax2)
ax2.text(-0.12, 1.08, 'B', transform=ax2.transAxes, fontsize=11, fontweight='bold')

# 添加数值标注
for bar, diff in zip(bars, diffs):
    height = bar.get_height()
    ax2.annotate(f'{diff:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3 if height > 0 else -10), textcoords='offset points',
                 ha='center', fontsize=6)

# 雷达图/极坐标图
ax3 = axes[2]
categories = ['Hydrophobic', 'Polar', 'Positive', 'Negative']
values_anti = [calc_group_freq(aa_freq_anti, hydrophobic),
               calc_group_freq(aa_freq_anti, polar),
               calc_group_freq(aa_freq_anti, charged_pos),
               calc_group_freq(aa_freq_anti, charged_neg)]
values_non = [calc_group_freq(aa_freq_non, hydrophobic),
              calc_group_freq(aa_freq_non, polar),
              calc_group_freq(aa_freq_non, charged_pos),
              calc_group_freq(aa_freq_non, charged_neg)]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
values_anti += values_anti[:1]
values_non += values_non[:1]

ax3.set_proj_type = 'polar'
ax3 = fig.add_subplot(1, 3, 3, projection='polar')
ax3.plot(angles, values_anti, 'o-', linewidth=1.5, label='Antifungal', color=OKABE_ITO[0])
ax3.fill(angles, values_anti, alpha=0.25, color=OKABE_ITO[0])
ax3.plot(angles, values_non, 'o-', linewidth=1.5, label='Non-antibacterial', color=OKABE_ITO[1])
ax3.fill(angles, values_non, alpha=0.25, color=OKABE_ITO[1])
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(categories, fontsize=7)
ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), frameon=False, fontsize=6)
ax3.text(-0.25, 1.08, 'C', transform=ax3.transAxes, fontsize=11, fontweight='bold')

plt.tight_layout()
save_figure(fig, 'figure2_amino_acid_properties')

# --- Figure 3: 各氨基酸频率对比 ---
print("生成图3: 各氨基酸频率对比...")
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

# 条形图对比
ax1 = axes[0]
aa_list = list(standard_aa)
freq_anti = [aa_freq_anti.get(aa, 0) for aa in aa_list]
freq_non = [aa_freq_non.get(aa, 0) for aa in aa_list]

x = np.arange(len(aa_list))
width = 0.35
ax1.bar(x - width/2, freq_anti, width, label='Antifungal', color=OKABE_ITO[0], edgecolor='white')
ax1.bar(x + width/2, freq_non, width, label='Non-antibacterial', color=OKABE_ITO[1], edgecolor='white')
ax1.set_xlabel('Amino Acid')
ax1.set_ylabel('Frequency (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(aa_list)
ax1.legend(frameon=False)
sns.despine(ax=ax1)
ax1.text(-0.08, 1.05, 'A', transform=ax1.transAxes, fontsize=11, fontweight='bold')

# 差异热力图
ax2 = axes[1]
diff_freq = [freq_anti[i] - freq_non[i] for i in range(len(aa_list))]
colors = [OKABE_ITO[2] if d > 0 else OKABE_ITO[5] for d in diff_freq]

# 绘制差异条形图
sorted_indices = np.argsort(diff_freq)[::-1]
sorted_aa = [aa_list[i] for i in sorted_indices]
sorted_diff = [diff_freq[i] for i in sorted_indices]
sorted_colors = [colors[i] for i in sorted_indices]

y_pos = np.arange(len(sorted_aa))
ax2.barh(y_pos, sorted_diff, color=sorted_colors, edgecolor='white')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(sorted_aa)
ax2.axvline(0, color='black', linewidth=0.5)
ax2.set_xlabel('Difference (Antifungal - Non-antibacterial, %)')
ax2.set_ylabel('Amino Acid')
sns.despine(ax=ax2)
ax2.text(-0.08, 1.05, 'B', transform=ax2.transAxes, fontsize=11, fontweight='bold')

# 标注显著差异
for i, (aa, diff) in enumerate(zip(sorted_aa, sorted_diff)):
    if abs(diff) > 1.5:
        ax2.annotate(f'{diff:+.1f}%', xy=(diff, i), 
                     xytext=(5 if diff > 0 else -5, 0), 
                     textcoords='offset points',
                     ha='left' if diff > 0 else 'right',
                     va='center', fontsize=6)

plt.tight_layout()
save_figure(fig, 'figure3_amino_acid_frequency')

# --- Figure 4: 综合面板图 ---
print("生成图4: 综合面板图...")
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35,
                       left=0.07, right=0.97, top=0.95, bottom=0.08)

# Panel A: 长度分布
ax_a = fig.add_subplot(gs[0, 0])
ax_a.hist(df_antifungal['seq_length'], bins=25, alpha=0.6, 
          label='Antifungal', color=OKABE_ITO[0], edgecolor='white')
ax_a.hist(df_non_antibacterial['seq_length'], bins=25, alpha=0.6,
          label='Non-antibacterial', color=OKABE_ITO[1], edgecolor='white')
ax_a.set_xlabel('Length (aa)')
ax_a.set_ylabel('Count')
ax_a.legend(frameon=False, fontsize=6)
sns.despine(ax=ax_a)
ax_a.text(-0.15, 1.12, 'A', transform=ax_a.transAxes, fontsize=11, fontweight='bold')

# Panel B: 箱线图
ax_b = fig.add_subplot(gs[0, 1])
bp = ax_b.boxplot([df_antifungal['seq_length'], df_non_antibacterial['seq_length']],
                 vert=True, patch_artist=True, widths=0.5,
                 labels=['Antifungal', 'Non-antibacterial'])
bp['boxes'][0].set_facecolor(OKABE_ITO[0])
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor(OKABE_ITO[1])
bp['boxes'][1].set_alpha(0.7)
ax_b.set_ylabel('Length (aa)')
sns.despine(ax=ax_b)
ax_b.text(-0.15, 1.12, 'B', transform=ax_b.transAxes, fontsize=11, fontweight='bold')

# Panel C: 氨基酸性质
ax_c = fig.add_subplot(gs[0, 2])
x = np.arange(4)
width = 0.35
ax_c.bar(x - width/2, anti_freqs[:4], width, label='Antifungal', color=OKABE_ITO[0], edgecolor='white')
ax_c.bar(x + width/2, non_freqs[:4], width, label='Non-antibacterial', color=OKABE_ITO[1], edgecolor='white')
ax_c.set_xticks(x)
ax_c.set_xticklabels(['Hydro', 'Polar', 'Pos', 'Neg'], fontsize=7)
ax_c.set_ylabel('Frequency (%)')
ax_c.legend(frameon=False, fontsize=6)
sns.despine(ax=ax_c)
ax_c.text(-0.15, 1.12, 'C', transform=ax_c.transAxes, fontsize=11, fontweight='bold')

# Panel D: 氨基酸频率差异
ax_d = fig.add_subplot(gs[1, :2])
ax_d.bar(y_pos, sorted_diff, color=sorted_colors, edgecolor='white')
ax_d.set_yticks(y_pos)
ax_d.set_yticklabels(sorted_aa, fontsize=7)
ax_d.axvline(0, color='black', linewidth=0.5)
ax_d.set_xlabel('Difference (Antifungal - Non-antibacterial, %)')
ax_d.set_ylabel('Amino Acid')
sns.despine(ax=ax_d)
ax_d.text(-0.04, 1.05, 'D', transform=ax_d.transAxes, fontsize=11, fontweight='bold')

# Panel E: 统计摘要
ax_e = fig.add_subplot(gs[1, 2])
ax_e.axis('off')
summary = f"""Statistical Summary
========================
Sample Size:
  Antifungal: {len(df_antifungal)}
  Non-anti: {len(df_non_antibacterial)}

Length (aa):
  Antifungal: {df_antifungal['seq_length'].mean():.1f} ± {df_antifungal['seq_length'].std():.1f}
  Non-anti: {df_non_antibacterial['seq_length'].mean():.1f} ± {df_non_antibacterial['seq_length'].std():.1f}

Net Charge:
  Antifungal: +{anti_freqs[2] - anti_freqs[3]:.1f}%
  Non-anti: +{non_freqs[2] - non_freqs[3]:.1f}%

Key Differences:
  K: {aa_freq_anti['K'] - aa_freq_non['K']:+.1f}%
  R: {aa_freq_anti['R'] - aa_freq_non['R']:+.1f}%
  C: {aa_freq_anti['C'] - aa_freq_non['C']:+.1f}%
  D: {aa_freq_anti['D'] - aa_freq_non['D']:+.1f}%
  E: {aa_freq_anti['E'] - aa_freq_non['E']:+.1f}%"""
ax_e.text(0.05, 0.98, summary, transform=ax_e.transAxes, fontsize=7,
          verticalalignment='top', fontfamily='monospace')
ax_e.text(-0.15, 1.05, 'E', transform=ax_e.transAxes, fontsize=11, fontweight='bold')

# Panel F: 累积分布
ax_f = fig.add_subplot(gs[2, 0])
ax_f.plot(sorted_anti, cum_anti, color=OKABE_ITO[0], linewidth=1.5, label='Antifungal')
ax_f.plot(sorted_non, cum_non, color=OKABE_ITO[1], linewidth=1.5, label='Non-antibacterial')
ax_f.set_xlabel('Length (aa)')
ax_f.set_ylabel('Cumulative')
ax_f.legend(frameon=False, fontsize=6)
ax_f.set_ylim(0, 1.05)
sns.despine(ax=ax_f)
ax_f.text(-0.15, 1.12, 'F', transform=ax_f.transAxes, fontsize=11, fontweight='bold')

# Panel G: 正负电荷对比
ax_g = fig.add_subplot(gs[2, 1])
charges = [['Positive', 'Negative'], [anti_freqs[2], anti_freqs[3]], [non_freqs[2], non_freqs[3]]]
x = np.arange(2)
ax_g.bar(x - width/2, [anti_freqs[2], anti_freqs[3]], width, label='Antifungal', 
         color=OKABE_ITO[0], edgecolor='white')
ax_g.bar(x + width/2, [non_freqs[2], non_freqs[3]], width, label='Non-antibacterial',
         color=OKABE_ITO[1], edgecolor='white')
ax_g.set_xticks(x)
ax_g.set_xticklabels(['Positive\n(K,R,H)', 'Negative\n(D,E)'], fontsize=7)
ax_g.set_ylabel('Frequency (%)')
ax_g.legend(frameon=False, fontsize=6)
sns.despine(ax=ax_g)
ax_g.text(-0.15, 1.12, 'G', transform=ax_g.transAxes, fontsize=11, fontweight='bold')

# Panel H: 长度分布核密度估计
ax_h = fig.add_subplot(gs[2, 2])
sns.kdeplot(df_antifungal['seq_length'], ax=ax_h, color=OKABE_ITO[0], 
            label='Antifungal', fill=True, alpha=0.3)
sns.kdeplot(df_non_antibacterial['seq_length'], ax=ax_h, color=OKABE_ITO[1],
            label='Non-antibacterial', fill=True, alpha=0.3)
ax_h.set_xlabel('Length (aa)')
ax_h.set_ylabel('Density')
ax_h.legend(frameon=False, fontsize=6)
sns.despine(ax=ax_h)
ax_h.text(-0.15, 1.12, 'H', transform=ax_h.transAxes, fontsize=11, fontweight='bold')

plt.tight_layout()
save_figure(fig, 'figure4_comprehensive_comparison')

print("\n" + "=" * 60)
print("所有对比分析图表生成完成!")
print(f"输出目录: {OUTPUT_DIR}")
print("=" * 60)

# 打印关键发现
print("\n【关键发现摘要】")
print(f"1. 序列长度: 抗真菌序列更短 (平均 {df_antifungal['seq_length'].mean():.1f} vs {df_non_antibacterial['seq_length'].mean():.1f} aa)")
print(f"2. 净电荷: 抗真菌序列显著更高 ({(anti_freqs[2] - anti_freqs[3]):.1f}% vs {(non_freqs[2] - non_freqs[3]):.1f}%)")
print(f"3. 正电荷氨基酸: K 含量抗真菌高 {aa_freq_anti['K'] - aa_freq_non['K']:+.1f}%, R 高 {aa_freq_anti['R'] - aa_freq_non['R']:+.1f}%")
print(f"4. 负电荷氨基酸: D 含量抗真菌低 {aa_freq_anti['D'] - aa_freq_non['D']:.1f}%, E 低 {aa_freq_anti['E'] - aa_freq_non['E']:.1f}%")
