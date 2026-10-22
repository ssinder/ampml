import pandas as pd


def extract_high_quality_samples():
    input_csv = "non_antimicrobial_peptides_strict_filtered.csv"
    output_csv = "non_antimicrobial_peptides_high_quality.csv"
    
    print(f"正在加载数据文件: {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8')
    print(f"加载数据完成，共 {len(df)} 条记录")
    
    print(f"\n正在筛选高质量样本（蛋白质水平证据）...")
    
    high_quality_df = df[df['protein_existence'].str.contains('Evidence at protein level', na=False)].copy()
    
    print(f"\n筛选结果:")
    print(f"  原始数据: {len(df)} 条")
    print(f"  高质量样本: {len(high_quality_df)} 条")
    print(f"  过滤掉: {len(df) - len(high_quality_df)} 条 ({(len(df) - len(high_quality_df))/len(df)*100:.1f}%)")
    
    print(f"\n正在保存高质量样本到: {output_csv}")
    high_quality_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"保存完成！")
    
    print(f"\n高质量样本统计:")
    print(f"\n1. 长度分布:")
    length_stats = high_quality_df['length'].describe()
    print(f"   平均长度: {length_stats['mean']:.1f} aa")
    print(f"   中位数: {length_stats['50%']:.1f} aa")
    print(f"   最小值: {length_stats['min']:.0f} aa")
    print(f"   最大值: {length_stats['max']:.0f} aa")
    
    print(f"\n2. 物种分布 (Top 10):")
    organism_counts = high_quality_df['organism'].value_counts().head(10)
    for organism, count in organism_counts.items():
        percentage = (count / len(high_quality_df)) * 100
        print(f"   {organism}: {count} 条 ({percentage:.1f}%)")
    
    print(f"\n3. 关键词分布 (Top 20):")
    all_keywords = []
    for keywords in high_quality_df['keywords'].dropna():
        if keywords:
            all_keywords.extend([kw.strip() for kw in keywords.split(';')])
    
    keyword_counts = pd.Series(all_keywords).value_counts().head(20)
    for keyword, count in keyword_counts.items():
        percentage = (count / len(high_quality_df)) * 100
        print(f"   {keyword}: {count} 条 ({percentage:.1f}%)")
    
    print(f"\n4. GO术语类别分布:")
    mf_count = (high_quality_df['go_molecular_function'].str.len() > 0).sum()
    bp_count = (high_quality_df['go_biological_process'].str.len() > 0).sum()
    cc_count = (high_quality_df['go_cellular_component'].str.len() > 0).sum()
    
    print(f"   分子功能: {mf_count} 条 ({mf_count/len(high_quality_df)*100:.1f}%)")
    print(f"   生物过程: {bp_count} 条 ({bp_count/len(high_quality_df)*100:.1f}%)")
    print(f"   细胞组分: {cc_count} 条 ({cc_count/len(high_quality_df)*100:.1f}%)")
    
    print(f"\n5. 蛋白质存在水平分布:")
    existence_counts = high_quality_df['protein_existence'].value_counts()
    for existence, count in existence_counts.items():
        percentage = (count / len(high_quality_df)) * 100
        print(f"   {existence}: {count} 条 ({percentage:.1f}%)")
    
    print(f"\n" + "=" * 60)
    print("高质量样本提取完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  {output_csv} - 高质量样本CSV数据")


if __name__ == "__main__":
    extract_high_quality_samples()
