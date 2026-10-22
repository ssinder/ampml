"""
提取CSV文件中的id和sequence列，保存为新的CSV文件
"""
import pandas as pd
import os

# 定义输入和输出文件路径
input_file = r"D:\trae\paper\data\sample\representative_sequences_filtered_no_x.csv"
output_file = r"D:\trae\paper\data\sample\id_sequence_extracted.csv"

# 读取原始CSV文件
print(f"正在读取文件: {input_file}")
df = pd.read_csv(input_file)

# 显示原始数据的列名和行数
print(f"原始数据列名: {df.columns.tolist()}")
print(f"原始数据行数: {len(df)}")

# 提取需要的列: representative_id 和 sequence
df_output = df[['representative_id', 'sequence']].copy()

# 保存到新的CSV文件
print(f"正在保存到: {output_file}")
df_output.to_csv(output_file, index=False)

print(f"提取完成！共提取 {len(df_output)} 条记录")
print(f"输出文件: {output_file}")

# 显示前几行数据作为验证
print("\n前5条记录预览:")
print(df_output.head())
