import pandas as pd
import os

data_dir = r'd:\trae\paper\data\all_antifungal'

print("=" * 70)
print("提取聚类代表序列")
print("=" * 70)

print("\n1. 读取聚类文件...")
cluster_file = os.path.join(data_dir, 'mmseq', 'merged_antifungal_le60_cluster.tsv')
cluster_df = pd.read_csv(cluster_file, sep='\t', header=None, names=['representative_id', 'member_id'])

print(f"   聚类文件总行数: {len(cluster_df)}")

print("\n2. 提取代表序列ID...")
representative_ids = cluster_df['representative_id'].unique()
print(f"   代表序列数量: {len(representative_ids)}")

print(f"   代表序列ID范围: {min(representative_ids)} - {max(representative_ids)}")

print("\n3. 读取原始数据集...")
original_df = pd.read_csv(os.path.join(data_dir, 'merged_antifungal_le60_final.csv'), encoding='latin1')
print(f"   原始数据集总行数: {len(original_df)}")

print("\n4. 提取代表序列...")

representative_sequences = []

for rep_id in representative_ids:
    row_index = rep_id - 1
    if row_index < len(original_df):
        row = original_df.iloc[row_index]
        representative_sequences.append({
            'Representative_ID': rep_id,
            'Sequence': row['Sequence'],
            'Sequence_Length': len(row['Sequence']),
            'Name': row['Name'] if pd.notna(row['Name']) else '',
            'DRAMP_ID': row['DRAMP_ID'] if pd.notna(row['DRAMP_ID']) else '',
            'Metadata_Sources': row['Metadata_Sources'] if pd.notna(row['Metadata_Sources']) else ''
        })
    else:
        print(f"   Warning: ID {rep_id} 超出数据范围")

print(f"   成功提取: {len(representative_sequences)} 个代表序列")

print("\n5. 创建代表序列DataFrame...")
rep_df = pd.DataFrame(representative_sequences)

print("\n6. 保存代表序列CSV...")
output_file = os.path.join(data_dir, 'mmseq', 'representative_sequences.csv')
rep_df.to_csv(output_file, index=False, encoding='utf-8')
print(f"   文件已保存: {output_file}")

print("\n7. 统计信息:")
print(f"   === 代表序列统计 ===")
print(f"   总代表序列数: {len(rep_df)}")
print(f"   平均序列长度: {rep_df['Sequence_Length'].mean():.2f}")
print(f"   最小序列长度: {rep_df['Sequence_Length'].min()}")
print(f"   最大序列长度: {rep_df['Sequence_Length'].max()}")

print(f"\n   === 元数据填充情况 ===")
print(f"   DRAMP_ID填充: {rep_df['DRAMP_ID'].notna().sum()}/{len(rep_df)} ({rep_df['DRAMP_ID'].notna().sum()/len(rep_df)*100:.2f}%)")
print(f"   Name填充: {rep_df['Name'].notna().sum()}/{len(rep_df)} ({rep_df['Name'].notna().sum()/len(rep_df)*100:.2f}%)")

print(f"\n   === 序列来源分布 ===")
source_counts = rep_df['Metadata_Sources'].value_counts()
for sources, count in source_counts.head(10).items():
    print(f"   {sources}: {count} 个序列")

print("\n8. 示例序列:")
print(f"   === 前10个代表序列 ===")
for i in range(min(10, len(rep_df))):
    row = rep_df.iloc[i]
    print(f"   {i+1}. ID: {row['Representative_ID']}, Seq: {row['Sequence'][:40]}... (长度: {row['Sequence_Length']})")

print("\n" + "=" * 70)
print("代表序列提取完成！")
print("=" * 70)

print(f"\n输出文件: {output_file}")
print(f"包含 {len(rep_df)} 个聚类代表序列")
print(f"聚类压缩率: {len(rep_df)}/{len(original_df)} = {len(rep_df)/len(original_df)*100:.2f}%")
