import pandas as pd

def csv_to_fasta(csv_file_path, fasta_file_path):
    """
    将过滤后的CSV文件转换为FASTA格式
    """
    print(f"正在读取CSV文件: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    print(f"数据集大小: {len(df)} 条记录")
    
    # 生成FASTA格式
    fasta_records = []
    
    for idx, row in df.iterrows():
        accession = row['accession']
        protein_name = row['protein_name']
        organism = row['organism']
        length = row['length']
        sequence = row['sequence']
        
        # 创建FASTA header
        header = f">{accession}|{protein_name}|{organism}|{length}aa"
        
        # 添加记录
        fasta_records.append(header)
        fasta_records.append(sequence)
    
    # 保存FASTA文件
    with open(fasta_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fasta_records))
    
    print(f"FASTA文件已保存到: {fasta_file_path}")
    print(f"文件大小: {len(fasta_records)} 行 (包含header和序列)")
    
    return fasta_records

def main():
    # 文件路径
    csv_file_path = r'd:\trae\paper\data\uniprot_peptide\non_antimicrobial_peptides_high_quality_complete_filtered.csv'
    fasta_file_path = r'd:\trae\paper\data\uniprot_peptide\non_antimicrobial_peptides_high_quality_complete_filtered.fasta'
    
    # 转换为FASTA格式
    fasta_records = csv_to_fasta(csv_file_path, fasta_file_path)
    
    print("\n转换完成！")
    print(f"\nFASTA格式示例（前3条记录）:")
    for i in range(min(6, len(fasta_records)), 0, 2):
        print(fasta_records[i-1])
        print(fasta_records[i][:80] + "..." if len(fasta_records[i]) > 80 else fasta_records[i])
        print()

if __name__ == '__main__':
    main()
