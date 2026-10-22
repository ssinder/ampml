import pandas as pd
import re

def parse_protein_names_from_md(md_file_path):
    """
    从llm_clean.md文件中解析需要过滤的蛋白质名称列表
    """
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    protein_names = set()
    
    # 匹配以#开头的标题行
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            continue
        if line == '':
            continue
        
        # 分割逗号分隔的蛋白质名称
        names = [name.strip() for name in line.split(',') if name.strip()]
        protein_names.update(names)
    
    return list(protein_names)

def filter_antimicrobial_peptides(csv_file_path, protein_names_to_filter, output_file_path):
    """
    根据蛋白质名称过滤掉潜在抗菌肽
    """
    print(f"正在读取CSV文件: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    print(f"原始数据集大小: {len(df)} 条记录")
    print(f"需要过滤的蛋白质名称数量: {len(protein_names_to_filter)}")
    
    # 创建过滤函数
    def should_filter(protein_name):
        if pd.isna(protein_name):
            return False
        
        protein_name_str = str(protein_name).strip()
        
        # 检查是否包含任何需要过滤的蛋白质名称
        for filter_name in protein_names_to_filter:
            if filter_name.lower() in protein_name_str.lower():
                return True
        
        return False
    
    # 应用过滤
    mask = df['protein_name'].apply(should_filter)
    filtered_df = df[~mask]
    
    print(f"过滤后数据集大小: {len(filtered_df)} 条记录")
    print(f"过滤掉的记录数: {len(df) - len(filtered_df)} 条记录")
    
    # 保存结果
    filtered_df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"过滤后的数据已保存到: {output_file_path}")
    
    # 显示一些统计信息
    print("\n=== 过滤统计 ===")
    print(f"过滤比例: {(len(df) - len(filtered_df)) / len(df) * 100:.2f}%")
    
    # 显示被过滤的蛋白质名称（前20个）
    filtered_records = df[mask]
    print(f"\n被过滤的蛋白质名称示例（前20个）:")
    for idx, row in filtered_records.head(20).iterrows():
        print(f"  - {row['protein_name']} (Accession: {row['accession']})")
    
    return filtered_df

def main():
    # 文件路径
    md_file_path = r'd:\trae\paper\data\uniprot_peptide\llm_clean.md'
    csv_file_path = r'd:\trae\paper\data\uniprot_peptide\non_antimicrobial_peptides_high_quality_complete.csv'
    output_file_path = r'd:\trae\paper\data\uniprot_peptide\non_antimicrobial_peptides_high_quality_complete_filtered.csv'
    
    # 解析需要过滤的蛋白质名称
    print("正在解析需要过滤的蛋白质名称...")
    protein_names_to_filter = parse_protein_names_from_md(md_file_path)
    print(f"解析到 {len(protein_names_to_filter)} 个需要过滤的蛋白质名称")
    
    # 显示一些示例
    print("\n需要过滤的蛋白质名称示例（前10个）:")
    for name in protein_names_to_filter[:10]:
        print(f"  - {name}")
    
    # 过滤数据
    print("\n开始过滤数据...")
    filtered_df = filter_antimicrobial_peptides(csv_file_path, protein_names_to_filter, output_file_path)
    
    print("\n过滤完成！")

if __name__ == '__main__':
    main()
