import pandas as pd
from collections import defaultdict


def export_representative_sequences_to_csv():
    cluster_file = "mmseq/non_antimicrobial_peptides_high_quality_complete_cluster.tsv"
    fasta_file = "mmseq/non_antimicrobial_peptides_high_quality_complete_rep_seq.fasta"
    output_csv = "mmseq/representative_sequences.csv"
    
    print("=" * 60)
    print("导出代表序列到CSV文件")
    print("=" * 60)
    
    print(f"\n正在加载聚类文件: {cluster_file}")
    cluster_df = pd.read_csv(cluster_file, sep='\t', header=None, names=['rep_seq_id', 'rep_seq_name', 'target_seq_id', 'target_seq_name'])
    print(f"聚类文件加载完成，共 {len(cluster_df)} 条记录")
    
    print(f"\n正在提取代表序列ID...")
    rep_seq_ids = cluster_df['rep_seq_id'].unique().tolist()
    print(f"找到 {len(rep_seq_ids)} 个唯一代表序列ID")
    
    print(f"\n正在读取FASTA文件: {fasta_file}")
    sequences = {}
    current_header = None
    current_seq = []
    
    with open(fasta_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences[current_header] = ''.join(current_seq)
                current_header = line[1:]
                current_seq = []
            else:
                if line:
                    current_seq.append(line)
        if current_header is not None:
            sequences[current_header] = ''.join(current_seq)
    
    print(f"FASTA文件加载完成，共 {len(sequences)} 条序列")
    
    print(f"\n正在匹配代表序列...")
    matched_sequences = []
    missing_ids = []
    
    for rep_id in rep_seq_ids:
        found = False
        for header, seq in sequences.items():
            if header.startswith(rep_id):
                simple_id = rep_id.split('|')[0]
                matched_sequences.append({
                    'representative_id': simple_id,
                    'sequence': seq
                })
                found = True
                break
        if not found:
            missing_ids.append(rep_id)
    
    print(f"成功匹配 {len(matched_sequences)} 条代表序列")
    if missing_ids:
        print(f"警告：{len(missing_ids)} 个代表序列ID在FASTA文件中未找到")
        print(f"缺失的ID: {missing_ids[:5]}...")
    
    print(f"\n正在导出到CSV文件: {output_csv}")
    output_df = pd.DataFrame(matched_sequences)
    output_df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"CSV文件导出完成！")
    
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    print(f"总代表序列数: {len(matched_sequences)}")
    print(f"输出文件: {output_csv}")
    print(f"列名: representative_id, sequence")
    
    print(f"\n前5条记录预览:")
    for i, row in output_df.head(5).iterrows():
        seq_preview = row['sequence'][:50] + '...' if len(row['sequence']) > 50 else row['sequence']
        print(f"  {row['representative_id']}: {seq_preview}")
    
    print("\n" + "=" * 60)
    print("导出完成！")
    print("=" * 60)


if __name__ == "__main__":
    export_representative_sequences_to_csv()
