import pandas as pd
import os
from collections import defaultdict

data_dir = r'd:\trae\paper\data\all_antifungal'

print("=" * 70)
print("基于改进的合并数据集生成过滤后的数据集")
print("=" * 70)

print("\n1. 读取改进的合并数据集...")
improved_df = pd.read_csv(os.path.join(data_dir, 'merged_antifungal_all_improved.csv'))
print(f"   总序列数: {len(improved_df)}")

files = {
    'Antifungal_amps.csv': 'Sequence',
    'antifungal.csv': 'Seq',
    'antifungal_peptides.csv': 'Sequence',
    'merged_data.csv': 'Seqence',
    'peptides_merged.csv': 'SEQUENCE'
}

standard_amino_acids = set('ARNDCEQGHILKMFPSTWYV')

def filter_standard_aa(sequence):
    for char in sequence:
        if char not in standard_amino_acids:
            return False
    return True

print("\n2. 读取原始文件并统计序列出现次数...")

seq_file_sets = defaultdict(set)

for filename, seq_col in files.items():
    filepath = os.path.join(data_dir, filename)
    print(f"   Reading {filename}...")
    df = pd.read_csv(filepath)
    
    if seq_col in df.columns:
        for seq in df[seq_col].dropna():
            seq_upper = seq.upper()
            if filter_standard_aa(seq):
                seq_file_sets[seq_upper].add(filename)

print(f"\n   总唯一序列数: {len(seq_file_sets)}")

print("\n3. 过滤至少在2个文件中出现的序列...")

sequences_in_2plus = [seq for seq, fileset in seq_file_sets.items() if len(fileset) >= 2]
print(f"   在2+文件中出现的序列: {len(sequences_in_2plus)}")

sequences_2plus_set = set(sequences_in_2plus)

filtered_2plus_df = improved_df[improved_df['Sequence'].isin(sequences_2plus_set)].copy()

print(f"   过滤后行数: {len(filtered_2plus_df)}")

occurrence_count = defaultdict(int)
for seq, fileset in seq_file_sets.items():
    if len(fileset) >= 2:
        occurrence_count[len(fileset)] += 1

print("\n4. 序列出现次数分布:")
print("   === 在2+文件中出现的序列分布 ===")
for count in sorted(occurrence_count.keys()):
    print(f"   出现在 {count} 个文件中的序列: {occurrence_count[count]}")

output_path = os.path.join(data_dir, 'merged_antifungal_2plus_improved.csv')
filtered_2plus_df.to_csv(output_path, index=False)
print(f"\n   CSV已保存到: {output_path}")

print("\n5. 统计结果:")
print("   === 总结 ===")
print(f"   改进合并数据集序列数: {len(improved_df)}")
print(f"   在2+文件中出现的唯一序列: {len(sequences_in_2plus)}")
print(f"   输出CSV行数: {len(filtered_2plus_df)}")
print(f"   移除的序列(仅1个文件): {len(improved_df) - len(filtered_2plus_df)}")
print(f"   保留率: {len(filtered_2plus_df)/len(improved_df)*100:.2f}%")

print("\n6. 元数据填充情况:")

for col in ['DRAMP_ID', 'Name']:
    if col in filtered_2plus_df.columns:
        filled = filtered_2plus_df[col].notna().sum()
        total = len(filtered_2plus_df)
        print(f"   {col}: {filled}/{total} ({filled/total*100:.2f}%)")

if 'Metadata_Sources' in filtered_2plus_df.columns:
    print("\n7. Metadata_Sources分布:")
    source_counts = filtered_2plus_df['Metadata_Sources'].value_counts()
    for sources, count in source_counts.head(10).items():
        print(f"   {sources}: {count} 个序列")

print("\n" + "=" * 70)
print("过滤完成！")
print("=" * 70)
print("改进点:")
print("1. 基于改进的合并数据集，包含完整的元数据")
print("2. 保留了所有来源的元数据信息")
print("3. 避免了元数据丢失")
print("4. 提供了更完整的数据集")
