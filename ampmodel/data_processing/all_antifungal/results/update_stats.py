# -*- coding: utf-8 -*-
"""
更新序列数量统计CSV - 包含MMSeqs聚类结果
"""
import pandas as pd
import os

# 设置工作目录
work_dir = r"D:\trae\solo\ampmodel\data_processing\all_antifungal"
results_dir = r"D:\trae\solo\ampmodel\data_processing\all_antifungal\results"

# 读取现有的统计数据
existing_csv = os.path.join(results_dir, "sequence_count_statistics.csv")

# 新数据
mmseq_rep_file = os.path.join(work_dir, "mmseq", "representative_sequences.csv")
df_mmseq = pd.read_csv(mmseq_rep_file, encoding="utf-8")
mmseq_count = len(df_mmseq)

print(f"MMSeqs代表性序列数量: {mmseq_count}")

# 创建新的统计表
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
        "5. 短肽过滤后-去冗余前(merged_antifungal_le60_final.csv)",
        "6. MMSeqs聚类代表性序列(representative_sequences.csv)"
    ],
    "Sequence_Count": [
        5708,
        1836,
        1813,
        562,
        6892,
        16811,
        7920,
        6921,
        5306,
        4758,
        mmseq_count
    ],
    "Notes": [
        "DRAMP抗真菌肽数据",
        "DRAMP抗真菌肽数据",
        "抗真菌肽数据",
        "合并数据",
        "肽类合并数据",
        "5个原始文件总计",
        "合并去重后",
        "标准化过滤后",
        "保留出现在≥2个文件的序列",
        "过滤长度≤60 aa的序列",
        "MMSeqs聚类，相似度100%去冗余"
    ]
})

# 保存更新后的序列数量统计表
stats_file = os.path.join(results_dir, "sequence_count_statistics.csv")
sequence_stats.to_csv(stats_file, index=False, encoding="utf-8-sig")
print(f"序列数量统计表已更新: {stats_file}")

print("\n更新后的序列数量统计表:")
print(sequence_stats.to_string(index=False))

# 计算去冗余率
original_count = 4758
dedup_count = mmseq_count
dedup_rate = (original_count - dedup_count) / original_count * 100

print(f"\n去冗余统计:")
print(f"去冗余前序列数: {original_count}")
print(f"去冗余后序列数: {dedup_count}")
print(f"去除冗余比例: {dedup_rate:.1f}%")