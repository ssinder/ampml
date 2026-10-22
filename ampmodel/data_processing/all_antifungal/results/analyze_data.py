# -*- coding: utf-8 -*-
"""
抗真菌肽数据处理 - 序列数量统计与分析脚本
用于分析各处理步骤的序列数量和数据库归属情况
"""
import pandas as pd
import os
import sys

# 设置工作目录
work_dir = r"D:\trae\solo\ampmodel\data_processing\all_antifungal"
results_dir = r"D:\trae\solo\ampmodel\data_processing\all_antifungal\results"

print("="*60)
print("Step 1: 分析各处理步骤的序列数量")
print("="*60)

def read_csv_with_encoding(filepath):
    """尝试多种编码读取CSV文件"""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "gbk"]
    for enc in encodings:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except:
            continue
    return pd.read_csv(filepath)

# 步骤1: 原始数据文件数量
dataset_dir = os.path.join(work_dir, "dataset")
original_files = {
    "antifungal.csv": os.path.join(dataset_dir, "antifungal.csv"),
    "Antifungal_amps.csv": os.path.join(dataset_dir, "Antifungal_amps.csv"),
    "antifungal_peptides.csv": os.path.join(dataset_dir, "antifungal_peptides.csv"),
    "merged_data.csv": os.path.join(dataset_dir, "merged_data.csv"),
    "peptides_merged.csv": os.path.join(dataset_dir, "peptides_merged.csv"),
}

print("\n原始数据文件统计:")
print("-"*40)
original_counts = {}
for name, path in original_files.items():
    df = read_csv_with_encoding(path)
    count = len(df)
    original_counts[name] = count
    print(f"{name}: {count} sequences")

total_original = sum(original_counts.values())
print(f"\n原始数据总计: {total_original} sequences")

# 步骤2: 合并后的文件
print("\n合并处理后的文件:")
print("-"*40)
merged_file = os.path.join(work_dir, "merged_antifungal_all_improved.csv")
df_merged = read_csv_with_encoding(merged_file)
merged_count = len(df_merged)
print(f"merged_antifungal_all_improved.csv: {merged_count} sequences")

# 步骤3: 标准化后的文件
print("\n标准化处理后的文件:")
print("-"*40)
standardized_file = os.path.join(work_dir, "merged_antifungal_all_improved_standardized.csv")
df_standardized = read_csv_with_encoding(standardized_file)
standardized_count = len(df_standardized)
print(f"merged_antifungal_all_improved_standardized.csv: {standardized_count} sequences")

# 步骤4: 2+文件过滤
print("\n2+文件过滤后的文件:")
print("-"*40)
two_plus_file = os.path.join(work_dir, "merged_antifungal_2plus_improved_standardized.csv")
df_two_plus = read_csv_with_encoding(two_plus_file)
two_plus_count = len(df_two_plus)
print(f"merged_antifungal_2plus_improved_standardized.csv: {two_plus_count} sequences")

# 步骤5: 短肽过滤(≤60 aa)
print("\n短肽过滤后的文件:")
print("-"*40)
le60_file = os.path.join(work_dir, "merged_antifungal_le60_final.csv")
try:
    df_le60 = pd.read_csv(le60_file, encoding="utf-8")
except UnicodeDecodeError:
    try:
        df_le60 = pd.read_csv(le60_file, encoding="latin-1")
    except:
        df_le60 = pd.read_csv(le60_file, encoding="gbk")
le60_count = len(df_le60)
print(f"merged_antifungal_le60_final.csv: {le60_count} sequences")

# 创建序列数量统计表
print("\n" + "="*60)
print("创建序列数量统计表...")
print("="*60)

sequence_stats = pd.DataFrame({
    "Step": [
        "1. 原始数据-antifungal.csv",
        "1. 原始数据-Antifungal_amps.csv",
        "1. 原始数据-antifungal_peptides.csv",
        "1. 原始数据-merged_data.csv",
        "1. 原始数据-peptides_merged.csv",
        "1. 原始数据总计",
        "2. 合并后(merged_antifungal_all_improved.csv)",
        "3. 标准化后(merged_antifungal_all_improved_standardized.csv)",
        "4. 2+文件过滤后(merged_antifungal_2plus_improved_standardized.csv)",
        "5. 短肽过滤后(merged_antifungal_le60_final.csv)"
    ],
    "Sequence_Count": [
        original_counts["antifungal.csv"],
        original_counts["Antifungal_amps.csv"],
        original_counts["antifungal_peptides.csv"],
        original_counts["merged_data.csv"],
        original_counts["peptides_merged.csv"],
        total_original,
        merged_count,
        standardized_count,
        two_plus_count,
        le60_count
    ]
})

# 保存序列数量统计表
stats_file = os.path.join(results_dir, "sequence_count_statistics.csv")
sequence_stats.to_csv(stats_file, index=False, encoding="utf-8-sig")
print(f"序列数量统计表已保存: {stats_file}")

print("\n序列数量统计表:")
print(sequence_stats.to_string(index=False))

# ========== 分析数据库归属情况 ==========
print("\n" + "="*60)
print("Step 2: 分析序列归属数据库情况")
print("="*60)

# 读取合并后的数据
df_merged = read_csv_with_encoding(merged_file)

# 检查是否有Metadata_Sources列
if "Metadata_Sources" in df_merged.columns:
    print("\n找到Metadata_Sources列，分析数据库归属...")
    
    # 解析数据库来源
    def parse_sources(source_str):
        if pd.isna(source_str):
            return []
        sources = str(source_str).split(",")
        return [s.strip() for s in sources if s.strip()]
    
    # 统计每个数据库的序列数量
    db_counts = {}
    for sources in df_merged["Metadata_Sources"].dropna():
        for source in parse_sources(sources):
            if source not in db_counts:
                db_counts[source] = 0
            db_counts[source] += 1
    
    print("\n数据库归属统计:")
    print("-"*40)
    for db, count in sorted(db_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{db}: {count} sequences")
    
    # 创建数据库归属统计表
    db_stats = pd.DataFrame({
        "Database": list(db_counts.keys()),
        "Sequence_Count": list(db_counts.values())
    })
    db_stats = db_stats.sort_values("Sequence_Count", ascending=False).reset_index(drop=True)
    
    # 保存数据库归属统计表
    db_stats_file = os.path.join(results_dir, "database_source_statistics.csv")
    db_stats.to_csv(db_stats_file, index=False, encoding="utf-8-sig")
    print(f"\n数据库归属统计表已保存: {db_stats_file}")
    
    # 保存完整的序列数据库归属详情
    detail_data = []
    for idx, row in df_merged.iterrows():
        seq = row.get("Sequence", "")
        sources = row.get("Metadata_Sources", "")
        detail_data.append({
            "Sequence": seq,
            "Database_Sources": sources
        })
    
    detail_df = pd.DataFrame(detail_data)
    detail_file = os.path.join(results_dir, "sequence_database_detail.csv")
    detail_df.to_csv(detail_file, index=False, encoding="utf-8-sig")
    print(f"序列数据库归属详情已保存: {detail_file}")

else:
    print("警告: 未找到Metadata_Sources列!")
    print("尝试查找其他可能的列...")
    print(f"可用列: {list(df_merged.columns)}")

print("\n" + "="*60)
print("分析完成!")
print("="*60)