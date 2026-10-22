#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析代表性序列的长度分布统计
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
csv_path = r'D:\trae\solo\ampmodel\data_processing\sample\representative_sequences_id_seq.csv'
df = pd.read_csv(csv_path)

# 打印原始数据信息
print("="*60)
print("数据基本信息:")
print(f"总序列数量: {len(df)}")
print(f"列名: {list(df.columns)}")
print("="*60)

# 计算序列长度
df['Length'] = df['Sequence'].apply(len)

# 基本统计信息
print("\n序列长度基本统计:")
print(f"最小长度: {df['Length'].min()}")
print(f"最大长度: {df['Length'].max()}")
print(f"平均长度: {df['Length'].mean():.2f}")
print(f"中位数长度: {df['Length'].median()}")
print(f"标准差: {df['Length'].std():.2f}")

# 长度分布统计表格
length_counts = df['Length'].value_counts().sort_index()
length_percent = (length_counts / len(df) * 100).round(2)

# 创建统计表格
stats_df = pd.DataFrame({
    'Length': length_counts.index,
    'Count': length_counts.values,
    'Percentage (%)': length_percent.values
})

# 添加累计计数和累计百分比
stats_df['Cumulative Count'] = stats_df['Count'].cumsum()
stats_df['Cumulative (%)'] = (stats_df['Cumulative Count'] / len(df) * 100).round(2)

print("\n" + "="*60)
print("序列长度分布统计表:")
print("="*60)
print(stats_df.to_string(index=False))

# 保存统计表格到CSV文件
output_csv = r'D:\trae\solo\ampmodel\data_processing\sample\sequence_length_distribution.csv'
stats_df.to_csv(output_csv, index=False)
print(f"\n统计表格已保存到: {output_csv}")

# 生成长度分布直方图
fig, ax = plt.subplots(figsize=(14, 6))

# 绘制直方图
min_len = df['Length'].min()
max_len = df['Length'].max()
bins = np.arange(min_len, max_len + 2) - 0.5
counts, bin_edges, patches = ax.hist(df['Length'], bins=bins, edgecolor='black', 
                                    color='steelblue', alpha=0.7, rwidth=0.8)

# 设置标题和标签 (英文)
ax.set_title('Distribution of Sequence Lengths', fontsize=14, fontweight='bold')
ax.set_xlabel('Sequence Length (amino acids)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)

# 添加网格线
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# 在每个柱子上添加数量标签
for count, patch in zip(counts, patches):
    if count > 0:
        ax.text(patch.get_x() + patch.get_width()/2, patch.get_height() + 0.5,
               str(int(count)), ha='center', va='bottom', fontsize=8)

plt.tight_layout()

# 保存图表
output_plot = r'D:\trae\solo\ampmodel\data_processing\sample\sequence_length_distribution.png'
plt.savefig(output_plot, dpi=150, bbox_inches='tight')
print(f"分布图已保存到: {output_plot}")

plt.close()

# 输出主要长度区间的统计
print("\n" + "="*60)
print("主要长度区间统计:")
print("="*60)

# 按照常见长度区间分组
bins_ranges = [(5, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 40), (41, 50), (51, 100)]
for start, end in bins_ranges:
    mask = (df['Length'] >= start) & (df['Length'] <= end)
    count = mask.sum()
    pct = count / len(df) * 100
    if count > 0:
        print(f"长度 {start:3d}-{end:3d}: {count:4d} 条 ({pct:5.2f}%)")

# 找出最常见的长度
top5_lengths = df['Length'].value_counts().head(5)
print("\n" + "="*60)
print("最常见的长度 (Top 5):")
print("="*60)
for length, count in top5_lengths.items():
    pct = count / len(df) * 100
    print(f"长度 {length}: {count} 条 ({pct:.2f}%)")

print("\n分析完成!")