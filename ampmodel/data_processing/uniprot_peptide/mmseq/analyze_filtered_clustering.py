import pandas as pd
from collections import defaultdict
import numpy as np

def parse_fasta(file_path):
    """
    手动解析FASTA文件
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split('|')[0]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences

def parse_fasta_with_header(file_path):
    """
    解析FASTA文件并保留header信息
    """
    sequences = {}
    current_id = None
    current_header = None
    current_seq = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = {
                        'header': current_header,
                        'sequence': ''.join(current_seq)
                    }
                current_header = line[1:]
                current_id = current_header.split('|')[0]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id is not None:
            sequences[current_id] = {
                'header': current_header,
                'sequence': ''.join(current_seq)
            }
    
    return sequences

def analyze_mmseq_clustering(cluster_file_path, rep_seq_file_path, all_seq_file_path):
    """
    分析MMseq2聚类结果
    """
    print("=" * 60)
    print("MMseq2聚类分析")
    print("=" * 60)
    
    # 读取聚类结果
    print(f"\n正在读取聚类结果文件: {cluster_file_path}")
    cluster_df = pd.read_csv(cluster_file_path, sep='\t', header=None, names=['rep_seq_id', 'member_seq_id'])
    
    # 读取代表序列
    print(f"正在读取代表序列文件: {rep_seq_file_path}")
    rep_sequences = parse_fasta_with_header(rep_seq_file_path)
    
    # 读取所有序列
    print(f"正在读取所有序列文件: {all_seq_file_path}")
    all_sequences = parse_fasta(all_seq_file_path)
    
    # 统计聚类信息
    total_sequences = len(all_sequences)
    total_representatives = len(rep_sequences)
    total_clusters = len(cluster_df['rep_seq_id'].unique())
    
    print(f"\n=== 聚类统计 ===")
    print(f"总序列数: {total_sequences}")
    print(f"代表序列数: {total_representatives}")
    print(f"聚类数: {total_clusters}")
    print(f"数据压缩率: {(1 - total_representatives/total_sequences) * 100:.2f}%")
    
    # 分析每个聚类的大小
    cluster_sizes = cluster_df.groupby('rep_seq_id').size()
    print(f"\n=== 聚类大小分布 ===")
    print(f"平均聚类大小: {cluster_sizes.mean():.2f}")
    print(f"中位数聚类大小: {cluster_sizes.median():.2f}")
    print(f"最小聚类大小: {cluster_sizes.min()}")
    print(f"最大聚类大小: {cluster_sizes.max()}")
    
    # 聚类大小分布
    size_distribution = cluster_sizes.value_counts().sort_index()
    print(f"\n聚类大小分布:")
    for size, count in size_distribution.head(20).items():
        print(f"  {size}个成员: {count}个聚类 ({count/total_clusters*100:.1f}%)")
    
    # 分析代表序列长度
    rep_lengths = [len(seq_info['sequence']) for seq_info in rep_sequences.values()]
    print(f"\n=== 代表序列长度分布 ===")
    print(f"平均长度: {np.mean(rep_lengths):.2f} aa")
    print(f"中位数长度: {np.median(rep_lengths):.2f} aa")
    print(f"最小长度: {min(rep_lengths)} aa")
    print(f"最大长度: {max(rep_lengths)} aa")
    
    # 检查未知氨基酸X
    sequences_with_x = []
    for seq_id, seq_info in rep_sequences.items():
        if 'X' in seq_info['sequence']:
            sequences_with_x.append(seq_id)
    
    print(f"\n=== 未知氨基酸X统计 ===")
    print(f"含有X的代表序列数: {len(sequences_with_x)}")
    print(f"含X比例: {len(sequences_with_x)/total_representatives*100:.2f}%")
    
    # 分析氨基酸组成
    from collections import Counter
    all_aa = ''.join([seq_info['sequence'] for seq_info in rep_sequences.values()])
    aa_counts = Counter(all_aa)
    
    print(f"\n=== 氨基酸组成（Top 20）===")
    for aa, count in aa_counts.most_common(20):
        print(f"  {aa}: {count} ({count/len(all_aa)*100:.2f}%)")
    
    return cluster_df, rep_sequences, all_sequences

def export_representative_sequences_csv(rep_sequences, output_csv_path):
    """
    导出代表序列为CSV格式
    """
    print(f"\n正在导出代表序列到CSV: {output_csv_path}")
    
    data = []
    for seq_id, seq_info in rep_sequences.items():
        header = seq_info['header']
        sequence = seq_info['sequence']
        
        # 解析header信息
        header_parts = header.split('|')
        accession = seq_id
        
        if len(header_parts) >= 3:
            protein_name = header_parts[1].strip()
            organism = header_parts[2].strip()
        else:
            protein_name = header
            organism = "Unknown"
        
        length = len(sequence)
        
        data.append({
            'representative_id': accession,
            'protein_name': protein_name,
            'organism': organism,
            'sequence': sequence,
            'length': length
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"已导出 {len(df)} 条代表序列")
    print(f"文件保存到: {output_csv_path}")
    
    return df

def filter_sequences_with_x(rep_sequences, output_csv_path):
    """
    过滤掉含有未知氨基酸X的序列
    """
    print(f"\n正在过滤含有X的序列...")
    
    filtered_data = []
    for seq_id, seq_info in rep_sequences.items():
        sequence = seq_info['sequence']
        if 'X' not in sequence:
            header = seq_info['header']
            
            header_parts = header.split('|')
            accession = seq_id
            
            if len(header_parts) >= 3:
                protein_name = header_parts[1].strip()
                organism = header_parts[2].strip()
            else:
                protein_name = header
                organism = "Unknown"
            
            filtered_data.append({
                'representative_id': accession,
                'protein_name': protein_name,
                'organism': organism,
                'sequence': sequence,
                'length': len(sequence)
            })
    
    df = pd.DataFrame(filtered_data)
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"过滤后剩余 {len(df)} 条代表序列")
    print(f"过滤比例: {(1 - len(df)/len(rep_sequences)) * 100:.2f}%")
    print(f"文件保存到: {output_csv_path}")
    
    return df

def analyze_rep_seq_features(rep_df, output_txt_path):
    """
    分析代表序列特征并生成报告
    """
    print(f"\n正在分析代表序列特征...")
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("代表序列特征分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 基本统计
        f.write("=== 基本统计 ===\n")
        f.write(f"代表序列总数: {len(rep_df)}\n")
        f.write(f"平均长度: {rep_df['length'].mean():.2f} aa\n")
        f.write(f"中位数长度: {rep_df['length'].median():.2f} aa\n")
        f.write(f"最小长度: {rep_df['length'].min()} aa\n")
        f.write(f"最大长度: {rep_df['length'].max()} aa\n")
        f.write(f"标准差: {rep_df['length'].std():.2f} aa\n\n")
        
        # 长度分布
        f.write("=== 长度分布 ===\n")
        length_bins = [0, 10, 20, 30, 40, 50, 60]
        length_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60']
        rep_df['length_bin'] = pd.cut(rep_df['length'], bins=length_bins, labels=length_labels)
        length_dist = rep_df['length_bin'].value_counts().sort_index()
        
        for bin_label, count in length_dist.items():
            f.write(f"{bin_label} aa: {count} ({count/len(rep_df)*100:.1f}%)\n")
        f.write("\n")
        
        # 物种分布（Top 20）
        f.write("=== 物种分布（Top 20）===\n")
        species_dist = rep_df['organism'].value_counts().head(20)
        for species, count in species_dist.items():
            f.write(f"{species}: {count} ({count/len(rep_df)*100:.1f}%)\n")
        f.write("\n")
        
        # 氨基酸组成
        f.write("=== 氨基酸组成 ===\n")
        from collections import Counter
        all_aa = ''.join(rep_df['sequence'].tolist())
        aa_counts = Counter(all_aa)
        
        for aa, count in aa_counts.most_common(20):
            f.write(f"{aa}: {count} ({count/len(all_aa)*100:.2f}%)\n")
        f.write("\n")
        
        # 前10条代表序列
        f.write("=== 前10条代表序列 ===\n")
        for idx, row in rep_df.head(10).iterrows():
            f.write(f"\n{row['representative_id']}|{row['protein_name']}|{row['organism']}|{row['length']}aa\n")
            f.write(f"{row['sequence']}\n")
    
    print(f"特征分析报告已保存到: {output_txt_path}")

def main():
    # 文件路径
    cluster_file_path = r'd:\trae\paper\data\uniprot_peptide\mmseq\non_antimicrobial_peptides_high_quality_complete_filtered_cluster.tsv'
    rep_seq_file_path = r'd:\trae\paper\data\uniprot_peptide\mmseq\non_antimicrobial_peptides_high_quality_complete_filtered_rep_seq.fasta'
    all_seq_file_path = r'd:\trae\paper\data\uniprot_peptide\mmseq\non_antimicrobial_peptides_high_quality_complete_filtered_all_seqs.fasta'
    
    # 输出文件路径
    rep_csv_path = r'd:\trae\paper\data\uniprot_peptide\mmseq\representative_sequences_filtered.csv'
    rep_no_x_csv_path = r'd:\trae\paper\data\uniprot_peptide\mmseq\representative_sequences_filtered_no_x.csv'
    analysis_txt_path = r'd:\trae\paper\data\uniprot_peptide\mmseq\representative_sequence_analysis_filtered.txt'
    
    # 分析聚类
    cluster_df, rep_sequences, all_sequences = analyze_mmseq_clustering(
        cluster_file_path, rep_seq_file_path, all_seq_file_path
    )
    
    # 导出代表序列CSV
    rep_df = export_representative_sequences_csv(rep_sequences, rep_csv_path)
    
    # 过滤含有X的序列
    rep_no_x_df = filter_sequences_with_x(rep_sequences, rep_no_x_csv_path)
    
    # 分析代表序列特征
    analyze_rep_seq_features(rep_no_x_df, analysis_txt_path)
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()
