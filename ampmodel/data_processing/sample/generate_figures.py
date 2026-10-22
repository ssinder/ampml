#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
抗菌肽序列对比分析 - 顶刊级别可视化 (扩展版)
抗真菌序列 vs 非抗菌序列
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

mpl.use('Agg')

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

# 数据加载
antifungal_path = 'd:/trae/paper/data/sample/representative_sequences_id_seq.csv'
non_antibacterial_path = 'd:/trae/paper/data/sample/id_sequence_extracted.csv'

df_antifungal = pd.read_csv(antifungal_path)
df_non_antibacterial = pd.read_csv(non_antibacterial_path)

if 'Sequence' in df_antifungal.columns:
    df_antifungal = df_antifungal.rename(columns={'Sequence': 'sequence'})

seq_anti = df_antifungal['sequence'].tolist()
seq_non = df_non_antibacterial['sequence'].tolist()

df_antifungal['seq_length'] = df_antifungal['sequence'].str.len()
df_non_antibacterial['seq_length'] = df_non_antibacterial['sequence'].str.len()

# 氨基酸性质
standard_aa = 'ACDEFGHIKLMNPQRSTVWY'

def calc_aa_composition(sequences):
    all_seqs = ''.join(sequences)
    aa_counts = {aa: all_seqs.count(aa) for aa in standard_aa}
    total = sum(aa_counts.values())
    return {aa: count / total * 100 for aa, count in aa_counts.items()}

aa_freq_anti = calc_aa_composition(seq_anti)
aa_freq_non = calc_aa_composition(seq_non)

# 氨基酸性质分组
hydrophobic = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']
polar = ['S', 'T', 'N', 'Q', 'Y', 'C']
charged_pos = ['K', 'R', 'H']
charged_neg = ['D', 'E']

def calc_group_freq(aa_freq, group):
    return sum(aa_freq.get(aa, 0) for aa in group)

# Kyte-Doolittle 疏水性表
aa_properties = {
    'A': {'hydrophobicity': 1.8, 'charge': 0},
    'R': {'hydrophobicity': -4.5, 'charge': 1},
    'N': {'hydrophobicity': -3.5, 'charge': 0},
    'D': {'hydrophobicity': -3.5, 'charge': -1},
    'C': {'hydrophobicity': 2.5, 'charge': 0},
    'Q': {'hydrophobicity': -3.5, 'charge': 0},
    'E': {'hydrophobicity': -3.5, 'charge': -1},
    'G': {'hydrophobicity': -0.4, 'charge': 0},
    'H': {'hydrophobicity': -3.2, 'charge': 0.1},
    'I': {'hydrophobicity': 4.5, 'charge': 0},
    'L': {'hydrophobicity': 3.8, 'charge': 0},
    'K': {'hydrophobicity': -3.9, 'charge': 1},
    'M': {'hydrophobicity': 1.9, 'charge': 0},
    'F': {'hydrophobicity': 2.8, 'charge': 0},
    'P': {'hydrophobicity': -1.6, 'charge': 0},
    'S': {'hydrophobicity': -0.8, 'charge': 0},
    'T': {'hydrophobicity': -0.7, 'charge': 0},
    'W': {'hydrophobicity': -0.9, 'charge': 0},
    'Y': {'hydrophobicity': -1.3, 'charge': 0},
    'V': {'hydrophobicity': 4.2, 'charge': 0}
}

def calc_hydrophobicity(sequence):
    if not sequence:
        return 0
    return sum(aa_properties.get(aa, {'hydrophobicity': 0})['hydrophobicity'] for aa in sequence) / len(sequence)

def calc_net_charge(sequence):
    if not sequence:
        return 0
    return sum(aa_properties.get(aa, {'charge': 0})['charge'] for aa in sequence)

def calc_shannon_entropy(sequence):
    if not sequence:
        return 0
    counts = Counter(sequence)
    total = len(sequence)
    entropy = 0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

# 计算序列属性
hydro_anti = [calc_hydrophobicity(seq) for seq in seq_anti]
hydro_non = [calc_hydrophobicity(seq) for seq in seq_non]
charge_anti = [calc_net_charge(seq) for seq in seq_anti]
charge_non = [calc_net_charge(seq) for seq in seq_non]
entropy_anti = [calc_shannon_entropy(seq) for seq in seq_anti]
entropy_non = [calc_shannon_entropy(seq) for seq in seq_non]

# 起始和结尾氨基酸
first_anti = Counter(seq[0] for seq in seq_anti if seq)
first_non = Counter(seq[0] for seq in seq_non if seq)
last_anti = Counter(seq[-1] for seq in seq_anti if seq)
last_non = Counter(seq[-1] for seq in seq_non if seq)

OUTPUT_DIR = 'd:/trae/paper/data/sample/figures'

def save_fig(fig, name, dpi=300):
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.pdf'), bbox_inches='tight', format='pdf')
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.png'), dpi=dpi, bbox_inches='tight', format='png')
    print(f"已保存: {name}.pdf/.png")
    plt.close(fig)

# ============================================================
# 已有图表 (保留)
# ============================================================

# 图1: 序列长度分布
print("生成图1: 序列长度分布...")
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.hist(df_antifungal['seq_length'], bins=20, alpha=0.6, 
        label=f'Antifungal (n={len(df_antifungal)})', color=OKABE_ITO[0], edgecolor='white')
ax.hist(df_non_antibacterial['seq_length'], bins=20, alpha=0.6,
        label=f'Non-antibacterial (n={len(df_non_antibacterial)})', color=OKABE_ITO[1], edgecolor='white')
ax.axvline(df_antifungal['seq_length'].mean(), color=OKABE_ITO[0], linestyle='--', linewidth=1.2)
ax.axvline(df_non_antibacterial['seq_length'].mean(), color=OKABE_ITO[1], linestyle='--', linewidth=1.2)
ax.set_xlabel('Sequence Length (aa)')
ax.set_ylabel('Count')
ax.legend(frameon=False, fontsize=7)
sns.despine(ax=ax)
save_fig(fig, 'fig1_length_distribution')

# 图2: 氨基酸性质对比
print("生成图2: 氨基酸性质对比...")
fig, ax = plt.subplots(figsize=(3.5, 2.5))
groups = ['Hydrophobic', 'Polar', 'Positive\n(K,R,H)', 'Negative\n(D,E)']
anti_freqs = [calc_group_freq(aa_freq_anti, hydrophobic),
              calc_group_freq(aa_freq_anti, polar),
              calc_group_freq(aa_freq_anti, charged_pos),
              calc_group_freq(aa_freq_anti, charged_neg)]
non_freqs = [calc_group_freq(aa_freq_non, hydrophobic),
             calc_group_freq(aa_freq_non, polar),
             calc_group_freq(aa_freq_non, charged_pos),
             calc_group_freq(aa_freq_non, charged_neg)]
x = np.arange(len(groups))
width = 0.35
ax.bar(x - width/2, anti_freqs, width, label='Antifungal', color=OKABE_ITO[0], edgecolor='white')
ax.bar(x + width/2, non_freqs, width, label='Non-antibacterial', color=OKABE_ITO[1], edgecolor='white')
ax.set_ylabel('Frequency (%)')
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=8)
ax.legend(frameon=False)
sns.despine(ax=ax)
save_fig(fig, 'fig2_amino_acid_properties')

# 图3: 氨基酸频率差异
print("生成图3: 氨基酸频率差异...")
fig, ax = plt.subplots(figsize=(3.5, 3))
diff_freq = [aa_freq_anti[aa] - aa_freq_non[aa] for aa in standard_aa]
sorted_indices = np.argsort(diff_freq)[::-1]
sorted_aa = [standard_aa[i] for i in sorted_indices]
sorted_diff = [diff_freq[i] for i in sorted_indices]
colors = [OKABE_ITO[2] if d > 0 else OKABE_ITO[5] for d in sorted_diff]
y_pos = np.arange(len(sorted_aa))
ax.barh(y_pos, sorted_diff, color=colors, edgecolor='white')
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_aa, fontsize=8)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('Difference (%, Antifungal - Non-antibacterial)')
ax.set_ylabel('Amino Acid')
sns.despine(ax=ax)
save_fig(fig, 'fig3_aa_frequency_difference')

# 图4: 净电荷对比
print("生成图4: 净电荷对比...")
fig, ax = plt.subplots(figsize=(3, 2.5))
net_anti = anti_freqs[2] - anti_freqs[3]
net_non = non_freqs[2] - non_freqs[3]
categories = ['Antifungal', 'Non-antibacterial']
values = [net_anti, net_non]
colors = [OKABE_ITO[0], OKABE_ITO[1]]
bars = ax.bar(categories, values, color=colors, edgecolor='white', width=0.5)
ax.set_ylabel('Net Charge (%)')
ax.set_ylim(0, max(values) * 1.3)
for bar, val in zip(bars, values):
    ax.annotate(f'+{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                ha='center', va='bottom', fontsize=9, fontweight='bold')
sns.despine(ax=ax)
save_fig(fig, 'fig4_net_charge')

# ============================================================
# 新增图表
# ============================================================

# 图5: 疏水性分布
print("生成图5: 疏水性分布...")
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.hist(hydro_anti, bins=25, alpha=0.6, label='Antifungal', color=OKABE_ITO[0], edgecolor='white', density=True)
ax.hist(hydro_non, bins=25, alpha=0.6, label='Non-antibacterial', color=OKABE_ITO[1], edgecolor='white', density=True)
ax.axvline(np.mean(hydro_anti), color=OKABE_ITO[0], linestyle='--', linewidth=1.2)
ax.axvline(np.mean(hydro_non), color=OKABE_ITO[1], linestyle='--', linewidth=1.2)
ax.set_xlabel('Hydrophobicity (Kyte-Doolittle)')
ax.set_ylabel('Density')
ax.legend(frameon=False, fontsize=7)
sns.despine(ax=ax)
save_fig(fig, 'fig5_hydrophobicity_distribution')

# 图6: 净电荷分布
print("生成图6: 净电荷分布...")
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.hist(charge_anti, bins=25, alpha=0.6, label='Antifungal', color=OKABE_ITO[0], edgecolor='white', density=True)
ax.hist(charge_non, bins=25, alpha=0.6, label='Non-antibacterial', color=OKABE_ITO[1], edgecolor='white', density=True)
ax.axvline(np.mean(charge_anti), color=OKABE_ITO[0], linestyle='--', linewidth=1.2)
ax.axvline(np.mean(charge_non), color=OKABE_ITO[1], linestyle='--', linewidth=1.2)
ax.set_xlabel('Net Charge')
ax.set_ylabel('Density')
ax.legend(frameon=False, fontsize=7)
sns.despine(ax=ax)
save_fig(fig, 'fig6_charge_distribution')

# 图7: Shannon熵 (序列复杂度)
print("生成图7: Shannon熵分布...")
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.hist(entropy_anti, bins=25, alpha=0.6, label='Antifungal', color=OKABE_ITO[0], edgecolor='white', density=True)
ax.hist(entropy_non, bins=25, alpha=0.6, label='Non-antibacterial', color=OKABE_ITO[1], edgecolor='white', density=True)
ax.set_xlabel('Shannon Entropy (bits)')
ax.set_ylabel('Density')
ax.legend(frameon=False, fontsize=7)
sns.despine(ax=ax)
save_fig(fig, 'fig7_shannon_entropy')

# 图8: 起始氨基酸 (N-terminal)
print("生成图8: 起始氨基酸频率...")
all_first = sorted(set(first_anti.keys()) | set(first_non.keys()))
first_anti_freq = [first_anti.get(aa, 0) / len(seq_anti) * 100 for aa in all_first]
first_non_freq = [first_non.get(aa, 0) / len(seq_non) * 100 for aa in all_first]
fig, ax = plt.subplots(figsize=(3.5, 2.5))
x = np.arange(len(all_first))
width = 0.35
ax.bar(x - width/2, first_anti_freq, width, label='Antifungal', color=OKABE_ITO[0], edgecolor='white')
ax.bar(x + width/2, first_non_freq, width, label='Non-antibacterial', color=OKABE_ITO[1], edgecolor='white')
ax.set_xlabel('N-terminal Amino Acid')
ax.set_ylabel('Frequency (%)')
ax.set_xticks(x)
ax.set_xticklabels(all_first, fontsize=8)
ax.legend(frameon=False, fontsize=7)
sns.despine(ax=ax)
save_fig(fig, 'fig8_n_terminal')

# 图9: 结尾氨基酸 (C-terminal)
print("生成图9: 结尾氨基酸频率...")
all_last = sorted(set(last_anti.keys()) | set(last_non.keys()))
last_anti_freq = [last_anti.get(aa, 0) / len(seq_anti) * 100 for aa in all_last]
last_non_freq = [last_non.get(aa, 0) / len(seq_non) * 100 for aa in all_last]
fig, ax = plt.subplots(figsize=(3.5, 2.5))
x = np.arange(len(all_last))
width = 0.35
ax.bar(x - width/2, last_anti_freq, width, label='Antifungal', color=OKABE_ITO[0], edgecolor='white')
ax.bar(x + width/2, last_non_freq, width, label='Non-antibacterial', color=OKABE_ITO[1], edgecolor='white')
ax.set_xlabel('C-terminal Amino Acid')
ax.set_ylabel('Frequency (%)')
ax.set_xticks(x)
ax.set_xticklabels(all_last, fontsize=8)
ax.legend(frameon=False, fontsize=7)
sns.despine(ax=ax)
save_fig(fig, 'fig9_c_terminal')

# 图10: 疏水性 vs 电荷散点图
print("生成图10: 疏水性 vs 电荷...")
fig, ax = plt.subplots(figsize=(3.5, 3))
ax.scatter(hydro_anti, charge_anti, alpha=0.3, s=10, c=OKABE_ITO[0], label='Antifungal')
ax.scatter(hydro_non, charge_non, alpha=0.3, s=10, c=OKABE_ITO[1], label='Non-antibacterial')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Hydrophobicity')
ax.set_ylabel('Net Charge')
ax.legend(frameon=False, fontsize=7)
sns.despine(ax=ax)
save_fig(fig, 'fig10_hydrophobicity_vs_charge')

# 图11: 详细氨基酸分类
print("生成图11: 详细氨基酸分类...")
categories = [
    ('Hydrophobic', hydrophobic),
    ('Polar', polar),
    ('Positive', charged_pos),
    ('Negative', charged_neg),
    ('Aromatic', ['F', 'Y', 'W']),
    ('Cysteine', ['C']),
    ('Glycine', ['G']),
    ('Proline', ['P'])
]
cat_names = [c[0] for c in categories]
anti_cat = [calc_group_freq(aa_freq_anti, c[1]) for c in categories]
non_cat = [calc_group_freq(aa_freq_non, c[1]) for c in categories]

fig, ax = plt.subplots(figsize=(3.5, 2.5))
x = np.arange(len(cat_names))
width = 0.35
ax.bar(x - width/2, anti_cat, width, label='Antifungal', color=OKABE_ITO[0], edgecolor='white')
ax.bar(x + width/2, non_cat, width, label='Non-antibacterial', color=OKABE_ITO[1], edgecolor='white')
ax.set_ylabel('Frequency (%)')
ax.set_xticks(x)
ax.set_xticklabels(cat_names, fontsize=7, rotation=30, ha='right')
ax.legend(frameon=False, fontsize=7)
sns.despine(ax=ax)
save_fig(fig, 'fig11_detailed_aa_categories')

print("\n" + "=" * 60)
print("所有图表生成完成!")
print(f"输出目录: {OUTPUT_DIR}")
print("=" * 60)
