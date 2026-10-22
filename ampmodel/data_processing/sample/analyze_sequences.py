#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
抗菌肽序列基本统计分析脚本
分析序列长度分布、氨基酸组成等基本特征
"""
import pandas as pd
import numpy as np

# 读取序列数据
df = pd.read_csv('d:/trae/paper/data/sample/id_sequence_extracted.csv')

print("=" * 50)
print("抗菌肽序列基本统计信息")
print("=" * 50)

# 基本信息
print(f"\n总序列数: {len(df)}")

# 序列长度分析
df['seq_length'] = df['sequence'].str.len()
print(f"\n序列长度统计:")
print(f"  最小长度: {df['seq_length'].min()}")
print(f"  最大长度: {df['seq_length'].max()}")
print(f"  平均长度: {df['seq_length'].mean():.2f}")
print(f"  中位数长度: {df['seq_length'].median():.2f}")
print(f"  标准差: {df['seq_length'].std():.2f}")

# 长度分布
print(f"\n序列长度分布:")
length_bins = [0, 10, 20, 30, 40, 50, 100, 200]
length_labels = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-100', '>100']
df['length_bin'] = pd.cut(df['seq_length'], bins=length_bins, labels=length_labels)
length_dist = df['length_bin'].value_counts().sort_index()
for bin_label, count in length_dist.items():
    print(f"  {bin_label}: {count} ({count/len(df)*100:.1f}%)")

# 氨基酸组成分析
all_sequences = ''.join(df['sequence'].tolist())
aa_counts = {}
for aa in all_sequences:
    aa_counts[aa] = aa_counts.get(aa, 0) + 1

# 按频率排序
sorted_aa = sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)

print(f"\n氨基酸组成分析 (Top 20):")
for aa, count in sorted_aa[:20]:
    freq = count / len(all_sequences) * 100
    print(f"  {aa}: {count} ({freq:.2f}%)")

# 统计非标准氨基酸
non_standard = {k: v for k, v in aa_counts.items() if k not in 'ACDEFGHIKLMNPQRSTVWY'}
if non_standard:
    print(f"\n非标准氨基酸:")
    for aa, count in non_standard.items():
        print(f"  {aa}: {count}")

# 保存统计结果
stats = {
    'total_sequences': len(df),
    'min_length': df['seq_length'].min(),
    'max_length': df['seq_length'].max(),
    'mean_length': df['seq_length'].mean(),
    'median_length': df['seq_length'].median(),
    'std_length': df['seq_length'].std()
}

import json
with open('d:/trae/paper/data/sample/sequence_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\n统计结果已保存到 sequence_stats.json")
