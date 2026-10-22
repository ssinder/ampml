import pandas as pd
import os
from collections import defaultdict

data_dir = r'd:\trae\paper\data\all_antifungal'

print("=" * 70)
print("验证最终数据集中的序列来源")
print("=" * 70)

print("\n1. 读取最终数据集...")
final_df = pd.read_csv(os.path.join(data_dir, 'merged_antifungal_le60_final.csv'), encoding='latin1')
print(f"   最终数据集总行数: {len(final_df)}")

print("\n2. 读取原始数据文件...")

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

seq_file_sets = defaultdict(set)

for filename, seq_col in files.items():
    filepath = os.path.join(data_dir, filename)
    print(f"   Reading {filename}...")
    df = pd.read_csv(filepath, encoding='latin1')
    
    if seq_col in df.columns:
        for seq in df[seq_col].dropna():
            seq_upper = str(seq).upper()
            if filter_standard_aa(seq_upper):
                seq_file_sets[seq_upper].add(filename)
    else:
        print(f"   Warning: Column '{seq_col}' not found in {filename}")

print(f"\n   总唯一序列数: {len(seq_file_sets)}")

print("\n3. 验证最终数据集中的序列来源...")

final_sequences = final_df['Sequence'].str.upper().tolist()

results = {
    'in_2plus': 0,
    'in_1file': 0,
    'not_found': 0,
    'source_distribution': defaultdict(int)
}

sequences_not_in_2plus = []
sequences_not_found = []
sequences_in_1file = []

for seq in final_sequences:
    if seq in seq_file_sets:
        sources = seq_file_sets[seq]
        num_sources = len(sources)
        results['source_distribution'][num_sources] += 1
        
        if num_sources >= 2:
            results['in_2plus'] += 1
        else:
            results['in_1file'] += 1
            sequences_in_1file.append((seq, sources))
    else:
        results['not_found'] += 1
        sequences_not_found.append(seq)

print("\n4. 验证结果:")
print("   === 序列来源统计 ===")
print(f"   在2+文件中出现的序列: {results['in_2plus']} ({results['in_2plus']/len(final_sequences)*100:.2f}%)")
print(f"   仅在1个文件中出现的序列: {results['in_1file']} ({results['in_1file']/len(final_sequences)*100:.2f}%)")
print(f"   未找到的序列: {results['not_found']} ({results['not_found']/len(final_sequences)*100:.2f}%)")

print("\n5. 序列来源分布:")
print("   === 按来源数量分布 ===")
for count in sorted(results['source_distribution'].keys()):
    print(f"   出现在 {count} 个文件中的序列: {results['source_distribution'][count]} ({results['source_distribution'][count]/len(final_sequences)*100:.2f}%)")

if results['not_found'] > 0:
    print(f"\n6. 未找到的序列示例:")
    for i, seq in enumerate(sequences_not_found[:5], 1):
        print(f"   {i}. {seq}")

if results['in_1file'] > 0:
    print(f"\n7. 仅在1个文件中出现的序列示例:")
    for i, (seq, sources) in enumerate(sequences_in_1file[:5], 1):
        print(f"   {i}. {seq[:50]}... (来源: {list(sources)})")

print("\n8. 检查Metadata_Sources列的一致性...")

if 'Metadata_Sources' in final_df.columns:
    print("   最终数据集包含Metadata_Sources列")
    
    mismatch_count = 0
    for seq in final_sequences[:100]:
        if seq in seq_file_sets:
            actual_sources = seq_file_sets[seq]
            
            matching_rows = final_df[final_df['Sequence'].str.upper() == seq]
            if len(matching_rows) > 0:
                metadata_sources = matching_rows.iloc[0]['Metadata_Sources']
                if pd.notna(metadata_sources):
                    metadata_source_list = [s.strip() for s in metadata_sources.split(',')]
                    
                    if set(actual_sources) != set(metadata_source_list):
                        mismatch_count += 1
    
    print(f"   检查了前100个序列，Metadata_Sources不匹配: {mismatch_count}")
else:
    print("   最终数据集不包含Metadata_Sources列")

print("\n9. 检查重复序列...")

duplicates = final_df[final_df.duplicated(subset=['Sequence'], keep=False)]
print(f"   重复序列数: {len(duplicates)}")

if len(duplicates) > 0:
    print(f"   重复序列示例:")
    for seq in duplicates['Sequence'].unique()[:5]:
        count = len(final_df[final_df['Sequence'] == seq])
        print(f"     {seq[:40]}... (出现 {count} 次)")

print("\n" + "=" * 70)
print("验证完成！")
print("=" * 70)

print("\n总结:")
print(f"  最终数据集总序列数: {len(final_sequences)}")
print(f"  符合2+文件要求的序列: {results['in_2plus']} ({results['in_2plus']/len(final_sequences)*100:.2f}%)")
print(f"  不符合要求的序列: {results['in_1file'] + results['not_found']} ({(results['in_1file'] + results['not_found'])/len(final_sequences)*100:.2f}%)")

if results['in_2plus'] == len(final_sequences):
    print(f"\n  ✅ 所有序列都符合要求（在至少2个文件中出现）！")
else:
    print(f"\n  ⚠️  发现 {results['in_1file'] + results['not_found']} 个序列不符合要求！")

print("\n建议:")
if results['in_2plus'] / len(final_sequences) >= 0.95:
    print("  数据集质量优秀，符合2+文件要求的序列占比 ≥ 95%")
elif results['in_2plus'] / len(final_sequences) >= 0.90:
    print("  数据集质量良好，符合2+文件要求的序列占比 ≥ 90%")
else:
    print("  建议重新检查数据集，不符合要求的序列占比较高")
