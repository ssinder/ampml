# -*- coding: utf-8 -*-
"""
绘制数据库归属Venn图 - 改进版
显示每个区域的序列数
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import numpy as np
import os

plt.rcParams['font.size'] = 10

# 设置工作目录
results_dir = r"D:\trae\solo\ampmodel\data_processing\all_antifungal\results"
work_dir = r"D:\trae\solo\ampmodel\data_processing\all_antifungal"

# 读取数据库归属数据
detail_file = os.path.join(results_dir, "sequence_database_detail.csv")
df = pd.read_csv(detail_file)

# 解析数据库来源
def parse_sources(source_str):
    if pd.isna(source_str):
        return []
    sources = str(source_str).split(",")
    return [s.strip() for s in sources if s.strip()]

# 为每条序列提取数据库集合
df["Sources_List"] = df["Database_Sources"].apply(parse_sources)

# 获取所有数据库
all_dbs = set()
for sources in df["Sources_List"]:
    all_dbs.update(sources)
all_dbs = sorted(list(all_dbs))
print(f"数据库列表: {all_dbs}")

# 计算每个可能的集合组合
from collections import Counter

set_combinations = Counter()
for sources in df["Sources_List"]:
    key = tuple(sorted(sources))
    set_combinations[key] += 1

print("\n数据库组合统计 (所有区域):")
for combo, count in sorted(set_combinations.items(), key=lambda x: x[1], reverse=True):
    print(f"{' ∩ '.join(combo)}: {count}")

# 为每个数据库创建序列集合
db_sequences = {db: set() for db in all_dbs}
for idx, row in df.iterrows():
    seq = row["Sequence"]
    for db in row["Sources_List"]:
        db_sequences[db].add(seq)

db_unique_counts = {db: len(db_sequences[db]) for db in all_dbs}

# ========== 改进的UpSet图 ==========
fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})

ax1 = axes[0]
combos = list(set_combinations.keys())
counts = [set_combinations[combo] for combo in combos]

# 按数量排序
sorted_indices = np.argsort(counts)[::-1]
combos = [combos[i] for i in sorted_indices]
counts = [counts[i] for i in sorted_indices]

# 创建条形图 - 水平方向更适合显示长标签
colors = plt.cm.Set3(np.linspace(0, 1, len(combos)))
bars = ax1.barh(range(len(combos)), counts, color=colors, edgecolor='gray')

# 设置标签
labels = [" ∩ ".join(combo) if len(combo) > 1 else combo[0] for combo in combos]
ax1.set_yticks(range(len(combos)))
ax1.set_yticklabels(labels, fontsize=9)

ax1.set_xlabel("Sequence Count", fontsize=12)
ax1.set_title("Database Source Distribution - UpSet Plot\n(Sequence Count by Database Combination)", fontsize=14, fontweight='bold')

# 在条形上添加数值
for bar, count in zip(bars, counts):
    ax1.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2, 
             str(count), ha='left', va='center', fontsize=9, fontweight='bold')

ax1.set_xlim(0, max(counts) * 1.2)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# 下部分：矩阵显示哪些数据库参与
ax2 = axes[1]
ax2.set_xlim(-0.5, len(combos)-0.5)
ax2.set_ylim(-0.5, len(all_dbs)-0.5)

# 绘制矩阵
for i, combo in enumerate(combos):
    for j, db in enumerate(all_dbs):
        if db in combo:
            ax2.add_patch(patches.Rectangle((i-0.4, len(all_dbs)-j-0.6), 0.8, 0.8, 
                                             facecolor='#2C3E50', edgecolor='gray'))
        else:
            ax2.add_patch(patches.Rectangle((i-0.4, len(all_dbs)-j-0.6), 0.8, 0.8, 
                                             facecolor='#ECF0F1', edgecolor='lightgray'))

ax2.set_yticks(range(len(all_dbs)))
ax2.set_yticklabels(all_dbs)
ax2.set_xticks([])
ax2.set_xlabel("Database Combinations (sorted by count)", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "database_upset_plot.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nUpSet图已保存: {os.path.join(results_dir, 'database_upset_plot.png')}")

# ========== 改进的饼图 ==========
fig, ax = plt.subplots(figsize=(12, 10))

colors = plt.cm.Set2(np.linspace(0, 1, len(all_dbs)))
wedges, texts, autotexts = ax.pie(
    db_unique_counts.values(), 
    labels=None,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    pctdistance=0.8,
    wedgeprops=dict(edgecolor='white', linewidth=2)
)

# 添加图例
legend_labels = [f"{db} ({count})" for db, count in db_unique_counts.items()]
ax.legend(wedges, legend_labels, title="Database (Sequence Count)", 
          loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)

# 添加圆环
centre_circle = plt.Circle((0,0), 0.65, fc='white', edgecolor='lightgray')
ax.add_artist(centre_circle)

ax.set_title("Database Source Distribution - Pie Chart\n(Unique Sequences per Database)", fontsize=14, fontweight='bold')

# 添加中心文字
ax.text(0, 0, f"Total\n{len(df)}", ha='center', va='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "database_pie_chart.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"饼图已保存: {os.path.join(results_dir, 'database_pie_chart.png')}")

# ========== 改进的堆叠条形图 ==========
fig, ax = plt.subplots(figsize=(14, 8))

# 准备数据
overlap_data = {}
for db in all_dbs:
    other_dbs = [d for d in all_dbs if d != db]
    unique_count = 0
    shared_with = {}
    for other_db in other_dbs:
        intersection = len(db_sequences[db] & db_sequences[other_db])
        if intersection > 0:
            shared_with[other_db] = intersection
    overlap_data[db] = {"unique": len(db_sequences[db]) - sum(shared_with.values()), 
                         "shared": shared_with}

db_names = list(overlap_data.keys())
x = np.arange(len(db_names))
width = 0.7

unique_counts = [overlap_data[db]["unique"] for db in db_names]
bottom = np.zeros(len(db_names))

# 共享部分用不同颜色
colors_db = plt.cm.Pastel1(np.linspace(0, 1, len(all_dbs)))

for j, db2 in enumerate(all_dbs):
    shared_counts = []
    for db1 in db_names:
        count = overlap_data[db1]["shared"].get(db2, 0)
        shared_counts.append(count)
    bars = ax.bar(x, shared_counts, width, bottom=bottom, label=f"Shared: {db2}", 
           color=colors_db[j], edgecolor='gray', linewidth=0.5, alpha=0.9)
    # 添加数值标签
    for k, (bar, count) in enumerate(zip(bars, shared_counts)):
        if count > 100:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                    str(count), ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    bottom += shared_counts

# 添加独有部分
unique_bars = ax.bar(x, unique_counts, width, bottom=bottom, label="Unique", 
       color='#2C3E50', edgecolor='gray', linewidth=0.5)
for k, (bar, count) in enumerate(zip(unique_bars, unique_counts)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bottom[k] + 50, 
            str(count), ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel("Database", fontsize=12)
ax.set_ylabel("Sequence Count", fontsize=12)
ax.set_title("Database Contribution - Stacked Bar Chart\n(Unique vs Shared Sequences)", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"{db}\n({db_unique_counts[db]})" for db in db_names], fontsize=9)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "database_stacked_bar.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"堆叠条形图已保存: {os.path.join(results_dir, 'database_stacked_bar.png')}")

# ========== 5路Venn示意图 - 改进版显示区域数量 ==========
fig, ax = plt.subplots(figsize=(14, 12))

# 使用5个圆的位置
angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
radius = 2.8

positions = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]

# 绘制圆
for i, (pos, db) in enumerate(zip(positions, all_dbs)):
    circle = Circle(pos, radius, alpha=0.2, facecolor=colors[i], edgecolor='gray', linewidth=2)
    ax.add_patch(circle)

# 计算并标注每个单独的圆和交集区域数量
# 获取区域数量
region_counts = {}
for combo, count in set_combinations.items():
    region_counts[combo] = count

# 标注每个数据库的总数量（在圆附近）
for i, (pos, db) in enumerate(zip(positions, all_dbs)):
    label_pos = (pos[0] * 1.5, pos[1] * 1.5)
    ax.annotate(f"{db}\n({db_unique_counts[db]})", label_pos, fontsize=9, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 标注2-数据库交集
two_way_intersections = []
for i, db1 in enumerate(all_dbs):
    for j, db2 in enumerate(all_dbs[i+1:], i+1):
        key = tuple(sorted([db1, db2]))
        count = region_counts.get(key, 0)
        if count > 0:
            mid_pos = ((positions[i][0] + positions[j][0])/2, (positions[i][1] + positions[j][1])/2)
            ax.annotate(str(count), mid_pos, fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7))

# 标注中心5路交集
center_count = region_counts.get(tuple(all_dbs), 0)
if center_count > 0:
    ax.annotate(f"5-way\n{center_count}", (0, 0), fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle='circle', facecolor='red', alpha=0.3), fontweight='bold')

ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title("5-Database Venn Diagram with Region Counts\n(Yellow numbers show 2-way intersections)", 
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "database_venn_schematic.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Venn示意图已保存: {os.path.join(results_dir, 'database_venn_schematic.png')}")

# ========== 创建详细的区域统计表 ==========
print("\n" + "="*60)
print("数据库归属详细统计表")
print("="*60)

# 创建详细的区域统计
region_stats = []
for combo, count in sorted(set_combinations.items(), key=lambda x: x[1], reverse=True):
    region_stats.append({
        "Database_Combination": " ∩ ".join(combo) if len(combo) > 1 else combo[0],
        "Sequence_Count": count,
        "Percentage": f"{count/len(df)*100:.1f}%"
    })

region_df = pd.DataFrame(region_stats)
region_file = os.path.join(results_dir, "database_region_statistics.csv")
region_df.to_csv(region_file, index=False, encoding="utf-8-sig")
print(f"区域统计表已保存: {region_file}")

print("\n各区域序列数统计:")
print(region_df.to_string(index=False))

print("\n所有图表绘制完成!")