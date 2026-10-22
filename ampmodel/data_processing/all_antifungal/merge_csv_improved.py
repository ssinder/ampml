import pandas as pd
import os

data_dir = r'd:\trae\paper\data\all_antifungal'

print("=" * 70)
print("改进的元数据合并脚本")
print("=" * 70)

files = {
    'Antifungal_amps.csv': {'seq_col': 'Sequence', 'name_col': 'Name'},
    'antifungal.csv': {'seq_col': 'Seq', 'name_col': 'Name'},
    'antifungal_peptides.csv': {'seq_col': 'Sequence', 'name_col': 'Name/Class'},
    'merged_data.csv': {'seq_col': 'Seqence', 'name_col': 'Title'},
    'peptides_merged.csv': {'seq_col': 'SEQUENCE', 'name_col': 'NAME'}
}

print("\n1. 读取所有文件...")

all_data = []
for filename, cols in files.items():
    filepath = os.path.join(data_dir, 'dataset', filename)
    print(f"Reading {filename}...")
    df = pd.read_csv(filepath, encoding='latin1')
    
    seq_col = cols['seq_col']
    name_col = cols['name_col']
    
    if seq_col in df.columns:
        df_renamed = df.copy()
        df_renamed = df_renamed.rename(columns={seq_col: 'Sequence'})
        
        if name_col in df.columns and name_col != 'Name':
            df_renamed = df_renamed.rename(columns={name_col: 'Name'})
        
        df_renamed['source_file'] = filename
        all_data.append(df_renamed)
        print(f"  Loaded {len(df_renamed)} rows from {filename}")
        print(f"  Name列: {name_col}")
    else:
        print(f"  Warning: Column '{seq_col}' not found in {filename}")

print("\n2. 合并所有数据...")

merged_df = pd.concat(all_data, ignore_index=True)
print(f"合并前总行数: {len(merged_df)}")

print("\n3. 按序列分组并合并元数据...")

sequence_groups = merged_df.groupby('Sequence')

print(f"发现 {len(sequence_groups)} 个唯一序列")

priority_order = [
    'Antifungal_amps.csv',
    'antifungal.csv',
    'antifungal_peptides.csv',
    'merged_data.csv',
    'peptides_merged.csv'
]

merged_rows = []
stats = {
    'total_sequences': len(sequence_groups),
    'single_source': 0,
    'multi_source': 0,
    'metadata_merged': 0
}

for sequence, group in sequence_groups:
    sources = group['source_file'].unique().tolist()
    
    if len(sources) == 1:
        stats['single_source'] += 1
        merged_row = group.iloc[0].copy()
        merged_row['Metadata_Sources'] = sources[0]
        merged_rows.append(merged_row)
    else:
        stats['multi_source'] += 1
        
        best_row = None
        best_score = -1
        
        for source in priority_order:
            if source in sources:
                source_rows = group[group['source_file'] == source]
                if len(source_rows) > 0:
                    candidate_row = source_rows.iloc[0].copy()
                    
                    score = 0
                    if not pd.isna(candidate_row.get('DRAMP_ID', pd.NA)):
                        score += 10
                    if not pd.isna(candidate_row.get('Name', pd.NA)):
                        score += 5
                    if not pd.isna(candidate_row.get('Source', pd.NA)):
                        score += 3
                    if not pd.isna(candidate_row.get('Activity', pd.NA)):
                        score += 2
                    
                    if score > best_score:
                        best_score = score
                        best_row = candidate_row
        
        if best_row is None:
            best_row = group.iloc[0].copy()
        
        for source in sources:
            if source != best_row['source_file']:
                source_rows = group[group['source_file'] == source]
                if len(source_rows) > 0:
                    other_row = source_rows.iloc[0]
                    
                    for col in other_row.index:
                        if col not in ['Sequence', 'source_file']:
                            if pd.isna(best_row.get(col, pd.NA)) and not pd.isna(other_row.get(col, pd.NA)):
                                best_row[col] = other_row[col]
                                stats['metadata_merged'] += 1
        
        best_row['Metadata_Sources'] = ', '.join(sorted(sources))
        merged_rows.append(best_row)

print(f"处理完成:")
print(f"  单一来源序列: {stats['single_source']}")
print(f"  多来源序列: {stats['multi_source']}")
print(f"  合并的元数据字段数: {stats['metadata_merged']}")

print("\n4. 生成最终合并数据...")

final_df = pd.DataFrame(merged_rows)
final_df = final_df.drop(columns=['source_file'], errors='ignore')

print(f"最终行数: {len(final_df)}")

print("\n5. 保存合并后的文件...")

output_path = os.path.join(data_dir, 'merged_antifungal_all_improved.csv')
final_df.to_csv(output_path, index=False, encoding='utf-8')

print(f"文件已保存到: {output_path}")

print("\n6. 统计结果...")

print(f"\n元数据填充情况:")
for col in ['DRAMP_ID', 'Name']:
    if col in final_df.columns:
        filled = final_df[col].notna().sum()
        total = len(final_df)
        print(f"  {col}: {filled}/{total} ({filled/total*100:.2f}%)")

print(f"\nMetadata_Sources分布:")
source_counts = final_df['Metadata_Sources'].value_counts()
for sources, count in source_counts.head(10).items():
    print(f"  {sources}: {count} 个序列")

print("\n" + "=" * 70)
print("改进的合并完成！")
print("=" * 70)
print("改进点:")
print("1. 保留了所有来源的元数据信息")
print("2. 按优先级选择最佳元数据")
print("3. 合并了不同来源的补充信息")
print("4. 添加了Metadata_Sources列记录来源")
print("5. 避免了元数据丢失")
