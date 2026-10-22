import pandas as pd


def export_to_fasta():
    input_csv = "non_antimicrobial_peptides_high_quality_complete.csv"
    output_fasta = "non_antimicrobial_peptides_high_quality_complete.fasta"
    
    print(f"正在加载数据文件: {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8')
    print(f"加载数据完成，共 {len(df)} 条记录")
    
    print(f"\n正在导出FASTA格式文件...")
    
    with open(output_fasta, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            accession = row['accession']
            sequence = row['sequence']
            protein_name = row.get('protein_name', '')
            organism = row.get('organism', '')
            length = row.get('length', 0)
            
            header = f">{accession}|{protein_name}|{organism}|{length}aa"
            f.write(header + '\n')
            f.write(sequence + '\n')
    
    print(f"FASTA文件已保存到: {output_fasta}")
    
    print(f"\n导出统计:")
    print(f"  总记录数: {len(df)}")
    print(f"  文件大小: {len(open(output_fasta, 'rb').read()) / 1024:.2f} KB")
    
    print(f"\n示例记录 (前3条):")
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        print(f"\n  记录 {idx + 1}:")
        print(f"    Accession: {row['accession']}")
        print(f"    Name: {row['protein_name']}")
        print(f"    Organism: {row['organism']}")
        print(f"    Length: {row['length']} aa")
        print(f"    Sequence: {row['sequence'][:50]}{'...' if len(row['sequence']) > 50 else ''}")
    
    print(f"\n" + "=" * 60)
    print("FASTA导出完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  {output_fasta} - FASTA格式文件")


if __name__ == "__main__":
    export_to_fasta()
