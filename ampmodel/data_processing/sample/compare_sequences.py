#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
抗菌肽序列对比统计分析
抗真菌序列 vs 非抗菌序列
"""
import pandas as pd
import numpy as np

# ============================================================
# 1. 加载数据
# ============================================================
antifungal_path = 'd:/trae/paper/data/sample/representative_sequences_id_seq.csv'
non_antibacterial_path = 'd:/trae/paper/data/sample/id_sequence_extracted.csv'

df_antifungal = pd.read_csv(antifungal_path)
df_non_antibacterial = pd.read_csv(non_antibacterial_path)

# 统一列名
if 'Sequence' in df_antifungal.columns:
    df_antifungal = df_antifungal.rename(columns={'Sequence': 'sequence'})
if 'sequence' in df_antifungal.columns:
    pass
else:
    print("抗真菌数据列名:", df_antifungal.columns.tolist())

if 'sequence' in df_non_antibacterial.columns:
    pass
else:
    print("非抗菌数据列名:", df_non_antibacterial.columns.tolist())

print("=" * 60)
print("抗菌肽序列对比统计分析")
print("=" * 60)

# ============================================================
# 2. 基本统计
# ============================================================
print(f"\n【数据概览】")
print(f"抗真菌序列数量: {len(df_antifungal)}")
print(f"非抗菌序列数量: {len(df_non_antibacterial)}")

# 序列长度计算
df_antifungal['seq_length'] = df_antifungal['sequence'].str.len()
df_non_antibacterial['seq_length'] = df_non_antibacterial['sequence'].str.len()

# ============================================================
# 3. 氨基酸组成分析
# ============================================================
standard_aa = 'ACDEFGHIKLMNPQRSTVWY'

def calc_aa_composition(sequences):
    """计算氨基酸组成"""
    all_seqs = ''.join(sequences.tolist())
    aa_counts = {aa: all_seqs.count(aa) for aa in standard_aa}
    total = sum(aa_counts.values())
    return {aa: count / total * 100 for aa, count in aa_counts.items()}

aa_freq_antifungal = calc_aa_composition(df_antifungal['sequence'])
aa_freq_non_antibacterial = calc_aa_composition(df_non_antibacterial['sequence'])

# 氨基酸性质分组
hydrophobic = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']
polar = ['S', 'T', 'N', 'Q', 'Y', 'C']
charged_pos = ['K', 'R', 'H']
charged_neg = ['D', 'E']

def calc_group_freq(aa_freq, group):
    return sum(aa_freq.get(aa, 0) for aa in group)

# ============================================================
# 4. 输出统计结果
# ============================================================
print(f"\n【序列长度统计】")
print(f"{'指标':<20} {'抗真菌序列':<15} {'非抗菌序列':<15}")
print("-" * 50)
print(f"{'最小长度':<18} {df_antifungal['seq_length'].min():<15} {df_non_antibacterial['seq_length'].min()}")
print(f"{'最大长度':<18} {df_antifungal['seq_length'].max():<15} {df_non_antibacterial['seq_length'].max()}")
print(f"{'平均长度':<18} {df_antifungal['seq_length'].mean():.2f}{'':<10} {df_non_antibacterial['seq_length'].mean():.2f}")
print(f"{'中位数长度':<18} {df_antifungal['seq_length'].median():.2f}{'':<10} {df_non_antibacterial['seq_length'].median():.2f}")
print(f"{'标准差':<18} {df_antifungal['seq_length'].std():.2f}{'':<10} {df_non_antibacterial['seq_length'].std():.2f}")

print(f"\n【氨基酸性质分布】")
print(f"{'性质':<20} {'抗真菌序列(%)':<15} {'非抗菌序列(%)':<15} {'差异':<10}")
print("-" * 60)

hydro_anti = calc_group_freq(aa_freq_antifungal, hydrophobic)
hydro_non = calc_group_freq(aa_freq_non_antibacterial, hydrophobic)
print(f"{'疏水性':<18} {hydro_anti:<15.2f} {hydro_non:<15.2f} {hydro_anti - hydro_non:+.2f}")

polar_anti = calc_group_freq(aa_freq_antifungal, polar)
polar_non = calc_group_freq(aa_freq_non_antibacterial, polar)
print(f"{'极性':<18} {polar_anti:<15.2f} {polar_non:<15.2f} {polar_anti - polar_non:+.2f}")

pos_anti = calc_group_freq(aa_freq_antifungal, charged_pos)
pos_non = calc_group_freq(aa_freq_non_antibacterial, charged_pos)
print(f"{'正电荷 (K,R,H)':<18} {pos_anti:<15.2f} {pos_non:<15.2f} {pos_anti - pos_non:+.2f}")

neg_anti = calc_group_freq(aa_freq_antifungal, charged_neg)
neg_non = calc_group_freq(aa_freq_non_antibacterial, charged_neg)
print(f"{'负电荷 (D,E)':<18} {neg_anti:<15.2f} {neg_non:<15.2f} {neg_anti - neg_non:+.2f}")

# 计算净电荷
net_charge_anti = pos_anti - neg_anti
net_charge_non = pos_non - neg_non
print(f"{'净电荷':<18} {net_charge_anti:<15.2f} {net_charge_non:<15.2f} {net_charge_anti - net_charge_non:+.2f}")

print(f"\n【各氨基酸频率对比】")
print(f"{'氨基酸':<8} {'抗真菌(%)':<12} {'非抗菌(%)':<12} {'差异':<10}")
print("-" * 42)
for aa in standard_aa:
    freq_anti = aa_freq_antifungal.get(aa, 0)
    freq_non = aa_freq_non_antibacterial.get(aa, 0)
    diff = freq_anti - freq_non
    marker = "↑" if diff > 1 else ("↓" if diff < -1 else "")
    print(f"{aa:<8} {freq_anti:<12.2f} {freq_non:<12.2f} {diff:+8.2f} {marker}")

# ============================================================
# 5. 保存统计数据
# ============================================================
import json

stats = {
    'antifungal': {
        'count': len(df_antifungal),
        'mean_length': float(df_antifungal['seq_length'].mean()),
        'median_length': float(df_antifungal['seq_length'].median()),
        'std_length': float(df_antifungal['seq_length'].std()),
        'min_length': int(df_antifungal['seq_length'].min()),
        'max_length': int(df_antifungal['seq_length'].max()),
        'aa_composition': aa_freq_antifungal,
        'hydrophobic': hydro_anti,
        'polar': polar_anti,
        'positive': pos_anti,
        'negative': neg_anti,
        'net_charge': net_charge_anti
    },
    'non_antibacterial': {
        'count': len(df_non_antibacterial),
        'mean_length': float(df_non_antibacterial['seq_length'].mean()),
        'median_length': float(df_non_antibacterial['seq_length'].median()),
        'std_length': float(df_non_antibacterial['seq_length'].std()),
        'min_length': int(df_non_antibacterial['seq_length'].min()),
        'max_length': int(df_non_antibacterial['seq_length'].max()),
        'aa_composition': aa_freq_non_antibacterial,
        'hydrophobic': hydro_non,
        'polar': polar_non,
        'positive': pos_non,
        'negative': neg_non,
        'net_charge': net_charge_non
    }
}

with open('d:/trae/paper/data/sample/comparison_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n统计数据已保存到 comparison_stats.json")
