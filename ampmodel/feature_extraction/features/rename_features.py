import csv
import os

# 输入和输出文件路径
input_file = 'results/all_features_combined.csv'
output_csv = 'results/all_features_combined_standard.csv'
output_txt = 'results/feature_name_mapping.txt'

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_dir, input_file)
output_csv_path = os.path.join(current_dir, output_csv)
output_txt_path = os.path.join(current_dir, output_txt)

print(f"处理文件: {input_path}")
print(f"输出CSV: {output_csv_path}")
print(f"输出TXT: {output_txt_path}")

# 读取CSV文件
with open(input_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # 读取表头
    rows = list(reader)    # 读取数据行

print(f"原始特征数量: {len(header) - 1}")  # 减去ID列

# 创建特征名称映射
name_mapping = {}
new_header = ['ID']  # 保留ID列

for feature in header[1:]:  # 从第二列开始处理特征
    if feature.startswith('iFeature_'):
        # 移除iFeature_前缀
        feature_without_prefix = feature[9:]
        # 进一步处理特征名称，移除AAC_、DPC_等前缀
        if '_' in feature_without_prefix:
            parts = feature_without_prefix.split('_')
            # 如果是AAC_开头，只保留氨基酸代码
            if parts[0] == 'AAC' and len(parts) == 2:
                new_name = parts[1]
            # 如果是DPC_开头，只保留二肽代码
            elif parts[0] == 'DPC' and len(parts) == 2:
                new_name = parts[1]
            # 其他特征保持原有格式
            else:
                new_name = feature_without_prefix
        else:
            new_name = feature_without_prefix
    elif feature.startswith('modlamp_'):
        # 移除modlamp_前缀
        new_name = feature[8:]
    elif feature.startswith('PseKRAAC_'):
        # 保持PseKRAAC格式不变
        new_name = feature
    else:
        # 其他特征保持不变
        new_name = feature
    
    name_mapping[feature] = new_name
    new_header.append(new_name)

print(f"新特征数量: {len(new_header) - 1}")

# 生成新的CSV文件
with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(new_header)
    for row in rows:
        writer.writerow(row)

print(f"新CSV文件已生成: {output_csv_path}")

# 生成名称对应txt文件
with open(output_txt_path, 'w', encoding='utf-8') as f:
    f.write("原始特征名称 -> 标准特征名称\n")
    f.write("=" * 50 + "\n")
    for old_name, new_name in name_mapping.items():
        f.write(f"{old_name} -> {new_name}\n")

print(f"名称对应文件已生成: {output_txt_path}")
print("处理完成！")