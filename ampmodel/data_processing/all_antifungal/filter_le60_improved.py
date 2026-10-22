import pandas as pd
import os

data_dir = r'd:\trae\paper\data\all_antifungal'

print("=" * 70)
print("短肽过滤与数据集比较")
print("=" * 70)

print("\n1. 读取标准化后的2+文件过滤数据集...")
standardized_df = pd.read_csv(os.path.join(data_dir, 'merged_antifungal_2plus_improved_standardized.csv'))
print(f"   总行数: {len(standardized_df)}")

print("\n2. 读取原来的短肽数据集...")
try:
    old_le60_df = pd.read_csv(os.path.join(data_dir, 'merged_antifungal_le60.csv'), encoding='utf-8')
except UnicodeDecodeError:
    old_le60_df = pd.read_csv(os.path.join(data_dir, 'merged_antifungal_le60.csv'), encoding='latin1')
print(f"   总行数: {len(old_le60_df)}")

print("\n3. 对标准化数据集进行短肽过滤(≤60 aa)...")

le60_standardized_df = standardized_df[standardized_df['Sequence'].str.len() <= 60].copy()
print(f"   过滤后行数: {len(le60_standardized_df)}")
print(f"   移除的序列(>60 aa): {len(standardized_df) - len(le60_standardized_df)}")
print(f"   保留率: {len(le60_standardized_df)/len(standardized_df)*100:.2f}%")

print("\n4. 序列长度分析...")

lengths = le60_standardized_df['Sequence'].str.len()
print(f"   平均长度: {lengths.mean():.2f}")
print(f"   最小长度: {lengths.min()}")
print(f"   最大长度: {lengths.max()}")
print(f"   中位数长度: {lengths.median():.2f}")

print("\n5. 比较新旧短肽数据集...")

old_sequences = set(old_le60_df['Sequence'].str.upper())
new_sequences = set(le60_standardized_df['Sequence'].str.upper())

print(f"   旧数据集唯一序列数: {len(old_sequences)}")
print(f"   新数据集唯一序列数: {len(new_sequences)}")

only_in_old = old_sequences - new_sequences
only_in_new = new_sequences - old_sequences
common_sequences = old_sequences & new_sequences

print(f"   仅在旧数据集中的序列: {len(only_in_old)}")
print(f"   仅在新数据集中的序列: {len(only_in_new)}")
print(f"   共同序列: {len(common_sequences)}")

print("\n6. 分析序列差异...")

if len(only_in_old) > 0:
    print(f"\n   仅在旧数据集中的序列示例:")
    for i, seq in enumerate(list(only_in_old)[:5], 1):
        print(f"     {i}. {seq[:50]}...")
    
    print(f"\n   分析这些序列的特征:")
    for seq in list(only_in_old)[:5]:
        matching_rows = old_le60_df[old_le60_df['Sequence'].str.upper() == seq]
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            print(f"     序列: {seq[:40]}...")
            print(f"       长度: {len(seq)}")
            print(f"       DRAMP_ID: {row['DRAMP_ID']}")
            print(f"       Name: {row['Name']}")

if len(only_in_new) > 0:
    print(f"\n   仅在新数据集中的序列示例:")
    for i, seq in enumerate(list(only_in_new)[:5], 1):
        print(f"     {i}. {seq[:50]}...")
    
    print(f"\n   分析这些序列的特征:")
    for seq in list(only_in_new)[:5]:
        matching_rows = le60_standardized_df[le60_standardized_df['Sequence'].str.upper() == seq]
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            print(f"     序列: {seq[:40]}...")
            print(f"       长度: {len(seq)}")
            print(f"       DRAMP_ID: {row['DRAMP_ID']}")
            print(f"       Name: {row['Name']}")
            if 'Metadata_Sources' in row:
                print(f"       Metadata_Sources: {row['Metadata_Sources']}")

print("\n7. 检查重复序列...")

old_duplicates = old_le60_df[old_le60_df.duplicated(subset=['Sequence'], keep=False)]
new_duplicates = le60_standardized_df[le60_standardized_df.duplicated(subset=['Sequence'], keep=False)]

print(f"   旧数据集重复序列数: {len(old_duplicates)}")
print(f"   新数据集重复序列数: {len(new_duplicates)}")

print("\n8. 比较元数据填充情况...")

print("   === 元数据填充对比 ===")
print(f"   {'列名':<25} {'旧数据集':<15} {'新数据集':<15} {'改进':<10}")
print(f"   {'-'*25} {'-'*15} {'-'*15} {'-'*10}")

for col in ['DRAMP_ID', 'Name']:
    if col in old_le60_df.columns and col in le60_standardized_df.columns:
        old_filled = old_le60_df[col].notna().sum()
        new_filled = le60_standardized_df[col].notna().sum()
        old_pct = old_filled / len(old_le60_df) * 100
        new_pct = new_filled / len(le60_standardized_df) * 100
        improvement = new_filled - old_filled
        print(f"   {col:<25} {old_filled:5d}/{len(old_le60_df):<5d} ({old_pct:5.2f}%) {new_filled:5d}/{len(le60_standardized_df):<5d} ({new_pct:5.2f}%) +{improvement:5d}")

print("\n9. 检查Metadata_Sources列...")

if 'Metadata_Sources' in le60_standardized_df.columns:
    print("   新数据集包含Metadata_Sources列")
    source_counts = le60_standardized_df['Metadata_Sources'].value_counts()
    
    print(f"\n   === Metadata_Sources分布 ===")
    total_sources = 0
    for sources, count in source_counts.items():
        num_sources = len(sources.split(','))
        total_sources += num_sources * count
        if count >= 100:
            print(f"   {sources}: {count} 个序列")
    
    avg_sources = total_sources / len(le60_standardized_df)
    print(f"\n   平均每个序列来自 {avg_sources:.2f} 个文件")
else:
    print("   旧数据集不包含Metadata_Sources列")

print("\n10. 检查元数据来源的多样性...")

if 'Metadata_Sources' in le60_standardized_df.columns:
    single_source = (le60_standardized_df['Metadata_Sources'].str.count(',') == 0).sum()
    multi_source = (le60_standardized_df['Metadata_Sources'].str.count(',') >= 1).sum()
    
    print(f"   === 来源多样性统计 ===")
    print(f"   单一来源序列: {single_source} ({single_source/len(le60_standardized_df)*100:.2f}%)")
    print(f"   多来源序列: {multi_source} ({multi_source/len(le60_standardized_df)*100:.2f}%)")

print("\n11. 检查序列长度分布...")

old_lengths = old_le60_df['Sequence'].str.len()
new_lengths = le60_standardized_df['Sequence'].str.len()

print("   === 序列长度分布对比 ===")
print(f"   旧数据集:")
print(f"     平均长度: {old_lengths.mean():.2f}")
print(f"     最小长度: {old_lengths.min()}")
print(f"     最大长度: {old_lengths.max()}")

print(f"\n   新数据集:")
print(f"     平均长度: {new_lengths.mean():.2f}")
print(f"     最小长度: {new_lengths.min()}")
print(f"     最大长度: {new_lengths.max()}")

print("\n12. 保存新的短肽数据集...")

output_path = os.path.join(data_dir, 'merged_antifungal_le60_improved_standardized.csv')
le60_standardized_df.to_csv(output_path, index=False)
print(f"   文件已保存: {output_path}")

print("\n" + "=" * 70)
print("短肽过滤与比较完成！")
print("=" * 70)

print("\n总结:")
print(f"  新数据集行数: {len(le60_standardized_df)}")
print(f"  旧数据集行数: {len(old_le60_df)}")
print(f"  行数差异: {len(le60_standardized_df) - len(old_le60_df)}")

print(f"\n  序列一致性: {len(common_sequences)} 个共同序列")
print(f"  旧数据集独有: {len(only_in_old)} 个序列")
print(f"  新数据集独有: {len(only_in_new)} 个序列")
print(f"  序列匹配率: {len(common_sequences)/len(old_sequences)*100:.2f}%")

if len(only_in_old) == 0 and len(only_in_new) == 0:
    print(f"\n  ✅ 序列完全一致！")
else:
    print(f"\n  ⚠️  发现序列不一致！")

print("\n关键改进:")
print("  1. 基于改进的合并数据集，包含完整元数据")
print("  2. 序列已标准化为大写格式")
print("  3. 过滤非标准氨基酸序列")
print("  4. 保留至少在2个文件中出现的序列")
print("  5. 保留所有来源的元数据信息")
print("  6. 添加Metadata_Sources列记录来源")
print("  7. 避免了元数据丢失")
print("  8. 提供了更完整的数据集")

print("\n推荐使用:")
print(f"  主要数据集: {output_path}")
