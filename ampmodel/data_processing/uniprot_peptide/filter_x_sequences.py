import pandas as pd

input_csv = r'D:\trae\paper\data\uniprot_peptide\mmseq\representative_sequences.csv'
output_csv = r'D:\trae\paper\data\uniprot_peptide\mmseq\representative_sequences_no_x.csv'

df = pd.read_csv(input_csv)

print(f"原始序列数: {len(df)}")

df_filtered = df[~df['sequence'].str.contains('X')]

print(f"过滤后序列数: {len(df_filtered)}")
print(f"去除的序列数: {len(df) - len(df_filtered)}")
print(f"去除比例: {(len(df) - len(df_filtered)) / len(df) * 100:.1f}%")

df_filtered.to_csv(output_csv, index=False)

print(f"\n已保存到: {output_csv}")
