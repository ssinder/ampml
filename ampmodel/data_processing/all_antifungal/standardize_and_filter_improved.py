import pandas as pd
import os
from collections import defaultdict

data_dir = r'd:\trae\paper\data\all_antifungal'

print("=" * 70)
print("改进数据集的标准化与2+文件过滤")
print("=" * 70)

print("\n1. 读取改进的合并数据集...")
improved_df = pd.read_csv(os.path.join(data_dir, 'merged_antifungal_all_improved.csv'))
print(f"   总行数: {len(improved_df)}")

print("\n2. 序列标准化...")

standard_amino_acids = set('ARNDCEQGHILKMFPSTWYV')

def filter_standard_aa(sequence):
    for char in sequence:
        if char not in standard_amino_acids:
            return False
    return True

improved_df['Sequence'] = improved_df['Sequence'].astype(str).str.upper()

print(f"   序列已转换为大写")

print("\n3. 过滤非标准氨基酸序列...")

valid_sequences = improved_df[improved_df['Sequence'].apply(filter_standard_aa)].copy()
invalid_count = len(improved_df) - len(valid_sequences)

print(f"   有效序列: {len(valid_sequences)}")
print(f"   无效序列: {invalid_count} ({invalid_count/len(improved_df)*100:.2f}%)")

if invalid_count > 0:
    invalid_sequences = improved_df[~improved_df['Sequence'].apply(filter_standard_aa)]
    print(f"\n   无效序列示例:")
    for i, seq in enumerate(invalid_sequences['Sequence'].head(5), 1):
        print(f"     {i}. {seq[:50]}...")

print("\n4. 读取原始文件并统计序列出现次数...")

files = {
    'Antifungal_amps.csv': 'Sequence',
    'antifungal.csv': 'Seq',
    'antifungal_peptides.csv': 'Sequence',
    'merged_data.csv': 'Seqence',
    'peptides_merged.csv': 'SEQUENCE'
}

seq_file_sets = defaultdict(set)

for filename, seq_col in files.items():
    filepath = os.path.join(data_dir, filename)
    print(f"   Reading {filename}...")
    df = pd.read_csv(filepath)
    
    if seq_col in df.columns:
        for seq in df[seq_col].dropna():
            seq_upper = str(seq).upper()
            if filter_standard_aa(seq_upper):
                seq_file_sets[seq_upper].add(filename)

print(f"\n   总唯一序列数: {len(seq_file_sets)}")

print("\n5. 过滤至少在2个文件中出现的序列...")

sequences_in_2plus = [seq for seq, fileset in seq_file_sets.items() if len(fileset) >= 2]
print(f"   在2+文件中出现的序列: {len(sequences_in_2plus)}")

sequences_2plus_set = set(sequences_in_2plus)

filtered_2plus_df = valid_sequences[valid_sequences['Sequence'].isin(sequences_2plus_set)].copy()

print(f"   过滤后行数: {len(filtered_2plus_df)}")

occurrence_count = defaultdict(int)
for seq, fileset in seq_file_sets.items():
    if len(fileset) >= 2:
        occurrence_count[len(fileset)] += 1

print("\n6. 序列出现次数分布:")
print("   === 在2+文件中出现的序列分布 ===")
for count in sorted(occurrence_count.keys()):
    print(f"   出现在 {count} 个文件中的序列: {occurrence_count[count]}")

print("\n7. 保存标准化后的完整数据集...")

output_filtered = os.path.join(data_dir, 'merged_antifungal_all_improved_standardized.csv')
valid_sequences.to_csv(output_filtered, index=False)
print(f"   文件已保存: {output_filtered}")

print("\n8. 保存2+文件过滤后的数据集...")

output_2plus = os.path.join(data_dir, 'merged_antifungal_2plus_improved_standardized.csv')
filtered_2plus_df.to_csv(output_2plus, index=False)
print(f"   文件已保存: {output_2plus}")

print("\n9. 统计结果:")

print("   === 标准化统计 ===")
print(f"   原始改进数据集: {len(improved_df)}")
print(f"   标准化后有效序列: {len(valid_sequences)}")
print(f"   过滤无效序列: {invalid_count} ({invalid_count/len(improved_df)*100:.2f}%)")
print(f"   保留率: {len(valid_sequences)/len(improved_df)*100:.2f}%")

print("\n   === 2+文件过滤统计 ===")
print(f"   标准化后序列: {len(valid_sequences)}")
print(f"   在2+文件中出现的唯一序列: {len(sequences_in_2plus)}")
print(f"   2+文件过滤后行数: {len(filtered_2plus_df)}")
print(f"   移除的序列(仅1个文件): {len(valid_sequences) - len(filtered_2plus_df)}")
print(f"   保留率: {len(filtered_2plus_df)/len(valid_sequences)*100:.2f}%")

print("\n10. 元数据填充情况:")

for col in ['DRAMP_ID', 'Name']:
    if col in filtered_2plus_df.columns:
        filled = filtered_2plus_df[col].notna().sum()
        total = len(filtered_2plus_df)
        print(f"   {col}: {filled}/{total} ({filled/total*100:.2f}%)")

if 'Metadata_Sources' in filtered_2plus_df.columns:
    print("\n11. Metadata_Sources分布:")
    source_counts = filtered_2plus_df['Metadata_Sources'].value_counts()
    for sources, count in source_counts.head(10).items():
        print(f"   {sources}: {count} 个序列")
    
    multi_source = (filtered_2plus_df['Metadata_Sources'].str.count(',') >= 1).sum()
    print(f"\n   多来源序列: {multi_source} ({multi_source/len(filtered_2plus_df)*100:.2f}%)")

print("\n12. 序列长度分析:")

lengths = filtered_2plus_df['Sequence'].str.len()
print(f"   平均长度: {lengths.mean():.2f}")
print(f"   最小长度: {lengths.min()}")
print(f"   最大长度: {lengths.max()}")
print(f"   中位数长度: {lengths.median():.2f}")

print("\n" + "=" * 70)
print("标准化与过滤完成！")
print("=" * 70)

print("\n生成的文件:")
print(f"  1. {output_filtered} - 标准化后的完整数据集")
print(f"  2. {output_2plus} - 标准化后的2+文件过滤数据集")

print("\n关键改进:")
print("  1. 基于改进的合并数据集，包含完整元数据")
print("  2. 序列标准化为大写格式")
print("  3. 过滤非标准氨基酸序列")
print("  4. 保留至少在2个文件中出现的序列")
print("  5. 保留所有来源的元数据信息")
print("  6. 添加Metadata_Sources列记录来源")
print("  7. 避免了元数据丢失")
print("  8. 提供了更完整的数据集")

print("\n推荐使用:")
print(f"  主要数据集: {output_2plus}")
print(f"  完整参考: {output_filtered}")
