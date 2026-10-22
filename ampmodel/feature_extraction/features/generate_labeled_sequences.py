# -*- coding: utf-8 -*-
"""
生成带标签的氨基酸序列CSV文件
"""

import pandas as pd
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data_processing" / "sample"
OUTPUT_DIR = PROJECT_ROOT / "feature_extraction" / "features" / "results"

# 读取CSV文件
df0 = pd.read_csv(DATA_DIR / "id_sequence_extracted.csv")
df1 = pd.read_csv(DATA_DIR / "representative_sequences_id_seq.csv")

print(f"CSV 0 before filtering: {len(df0)}")
print(f"CSV 1 before filtering: {len(df1)}")

# 过滤掉包含非标准氨基酸的序列和长度小于10的序列
non_standard = set("UBZXJWO")

def is_valid_sequence(seq):
    if any(aa in non_standard for aa in seq.upper()):
        return False
    if len(seq) < 10:
        return False
    return True

df0_valid = df0[df0["sequence"].apply(is_valid_sequence)].copy()
df1_valid = df1[df1["Sequence"].apply(is_valid_sequence)].copy()

print(f"CSV 0 after filtering: {len(df0_valid)}")
print(f"CSV 1 after filtering: {len(df1_valid)}")

# 创建序列DataFrame
sequences = []

# 添加标签0的序列
for idx, row in df0_valid.iterrows():
    sequences.append({
        "ID": f"L0_{row['representative_id']}",
        "Sequence": row["sequence"],
        "Label": 0
    })

# 添加标签1的序列
for idx, row in df1_valid.iterrows():
    sequences.append({
        "ID": f"L1_{row['Representative_ID']}",
        "Sequence": row["Sequence"],
        "Label": 1
    })

# 保存为CSV
df_seq = pd.DataFrame(sequences)
output_file = OUTPUT_DIR / "sequences_with_labels.csv"
df_seq.to_csv(output_file, index=False)

print(f"\nSaved to: {output_file}")
print(f"Total sequences: {len(df_seq)}")
print(f"Label 0: {(df_seq['Label'] == 0).sum()}")
print(f"Label 1: {(df_seq['Label'] == 1).sum()}")
print(f"\nFirst 5 rows:")
print(df_seq.head())
