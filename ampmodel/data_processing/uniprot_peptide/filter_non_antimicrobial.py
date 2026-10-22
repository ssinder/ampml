import pandas as pd
import numpy as np
from typing import List, Set, Dict


class NonAntimicrobialPeptideFilter:
    def __init__(self):
        self.antimicrobial_go_terms = {
            'GO:0042742', 'GO:0031640', 'GO:0006952', 'GO:0090729',
            'GO:0045087', 'GO:0050832', 'GO:0050829', 'GO:0050830'
        }
        
        self.antimicrobial_keywords = {
            'Antimicrobial', 'Defensin', 'Antibiotic', 'Bacteriocin',
            'Toxin', 'Antiviral', 'Allergen', 'Antifungal',
            'Antiparasitic', 'Immunity', 'Innate immunity', 'Defense',
            'Antibiotic resistance', 'Antibiotic biosynthesis', 'Plant defense',
            'Amphibian defense peptide', 'Insect defense', 'Fungicide',
            'Neurotoxin', 'Cardiotoxin', 'Hemorrhagic toxin', 'Myotoxin'
        }
        
        self.extracellular_go_terms = {'GO:0005576', 'GO:0005615'}

    def load_data(self, csv_file: str) -> pd.DataFrame:
        print(f"正在加载数据文件: {csv_file}")
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"加载数据完成，共 {len(df)} 条记录")
        return df

    def has_antimicrobial_go_terms(self, go_terms: str) -> bool:
        if pd.isna(go_terms) or go_terms == '':
            return False
        return any(go_id in go_terms for go_id in self.antimicrobial_go_terms)

    def has_antimicrobial_keywords(self, keywords: str) -> bool:
        if pd.isna(keywords) or keywords == '':
            return False
        keywords_lower = keywords.lower()
        return any(kw.lower() in keywords_lower for kw in self.antimicrobial_keywords)

    def is_extracellular(self, go_terms: str) -> bool:
        if pd.isna(go_terms) or go_terms == '':
            return False
        return any(go_id in go_terms for go_id in self.extracellular_go_terms)

    def calculate_antimicrobial_score(self, row: pd.Series) -> float:
        score = 0.0
        
        go_terms = row.get('go_terms', '')
        keywords = row.get('keywords', '')
        
        if self.has_antimicrobial_go_terms(go_terms):
            score += 3.0
        
        if self.has_antimicrobial_keywords(keywords):
            score += 2.0
        
        if self.is_extracellular(go_terms):
            score += 1.0
        
        return score

    def filter_peptides(self, df: pd.DataFrame, max_score: float = 0.0) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("开始筛选非抗菌肽")
        print("=" * 60)
        
        df_copy = df.copy()
        
        print("\n正在计算抗菌评分...")
        df_copy['antimicrobial_score'] = df_copy.apply(self.calculate_antimicrobial_score, axis=1)
        
        print("\n评分统计:")
        score_counts = df_copy['antimicrobial_score'].value_counts().sort_index()
        for score, count in score_counts.items():
            percentage = (count / len(df_copy)) * 100
            print(f"  评分 {score:.1f}: {count} 条 ({percentage:.1f}%)")
        
        print(f"\n筛选条件: 抗菌评分 <= {max_score}")
        filtered_df = df_copy[df_copy['antimicrobial_score'] <= max_score].copy()
        
        print(f"\n筛选结果:")
        print(f"  原始数据: {len(df)} 条")
        print(f"  筛选后数据: {len(filtered_df)} 条")
        print(f"  过滤掉: {len(df) - len(filtered_df)} 条 ({(len(df) - len(filtered_df))/len(df)*100:.1f}%)")
        
        return filtered_df

    def analyze_filtered_data(self, df: pd.DataFrame) -> Dict:
        print("\n" + "=" * 60)
        print("筛选结果分析")
        print("=" * 60)
        
        analysis = {}
        
        print("\n1. 长度分布:")
        length_stats = df['length'].describe()
        print(f"   平均长度: {length_stats['mean']:.1f} aa")
        print(f"   中位数: {length_stats['50%']:.1f} aa")
        print(f"   最小值: {length_stats['min']:.0f} aa")
        print(f"   最大值: {length_stats['max']:.0f} aa")
        analysis['length_stats'] = length_stats.to_dict()
        
        print("\n2. 物种分布 (Top 10):")
        organism_counts = df['organism'].value_counts().head(10)
        for organism, count in organism_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {organism}: {count} 条 ({percentage:.1f}%)")
        analysis['organism_distribution'] = organism_counts.to_dict()
        
        print("\n3. 关键词分布 (Top 20):")
        all_keywords = []
        for keywords in df['keywords'].dropna():
            if keywords:
                all_keywords.extend([kw.strip() for kw in keywords.split(';')])
        
        keyword_counts = pd.Series(all_keywords).value_counts().head(20)
        for keyword, count in keyword_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {keyword}: {count} 条 ({percentage:.1f}%)")
        analysis['keyword_distribution'] = keyword_counts.to_dict()
        
        print("\n4. GO术语类别分布:")
        mf_count = (df['go_molecular_function'].str.len() > 0).sum()
        bp_count = (df['go_biological_process'].str.len() > 0).sum()
        cc_count = (df['go_cellular_component'].str.len() > 0).sum()
        
        print(f"   分子功能: {mf_count} 条 ({mf_count/len(df)*100:.1f}%)")
        print(f"   生物过程: {bp_count} 条 ({bp_count/len(df)*100:.1f}%)")
        print(f"   细胞组分: {cc_count} 条 ({cc_count/len(df)*100:.1f}%)")
        
        analysis['go_category_distribution'] = {
            'molecular_function': int(mf_count),
            'biological_process': int(bp_count),
            'cellular_component': int(cc_count)
        }
        
        print("\n5. 蛋白质存在水平分布:")
        existence_counts = df['protein_existence'].value_counts()
        for existence, count in existence_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {existence}: {count} 条 ({percentage:.1f}%)")
        analysis['protein_existence_distribution'] = existence_counts.to_dict()
        
        return analysis

    def save_filtered_data(self, df: pd.DataFrame, output_file: str):
        print(f"\n正在保存筛选结果到: {output_file}")
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"保存完成！")

    def save_analysis_report(self, analysis: Dict, output_file: str):
        print(f"\n正在保存分析报告到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 非抗菌肽初筛结果分析报告\n\n")
            f.write(f"## 数据概览\n\n")
            f.write(f"- 筛选条件: 抗菌评分 <= 0.0\n")
            f.write(f"- 筛选结果数量: {len(analysis.get('length_stats', {}))} 条\n\n")
            
            f.write(f"## 长度分布\n\n")
            length_stats = analysis.get('length_stats', {})
            f.write(f"- 平均长度: {length_stats.get('mean', 0):.1f} aa\n")
            f.write(f"- 中位数: {length_stats.get('50%', 0):.1f} aa\n")
            f.write(f"- 最小值: {length_stats.get('min', 0):.0f} aa\n")
            f.write(f"- 最大值: {length_stats.get('max', 0):.0f} aa\n\n")
            
            f.write(f"## 物种分布 (Top 10)\n\n")
            organism_dist = analysis.get('organism_distribution', {})
            for organism, count in list(organism_dist.items())[:10]:
                f.write(f"- {organism}: {count} 条\n")
            
            f.write(f"\n## 关键词分布 (Top 20)\n\n")
            keyword_dist = analysis.get('keyword_distribution', {})
            for keyword, count in list(keyword_dist.items())[:20]:
                f.write(f"- {keyword}: {count} 条\n")
            
            f.write(f"\n## GO术语类别分布\n\n")
            go_dist = analysis.get('go_category_distribution', {})
            f.write(f"- 分子功能: {go_dist.get('molecular_function', 0)} 条\n")
            f.write(f"- 生物过程: {go_dist.get('biological_process', 0)} 条\n")
            f.write(f"- 细胞组分: {go_dist.get('cellular_component', 0)} 条\n")
            
            f.write(f"\n## 蛋白质存在水平分布\n\n")
            existence_dist = analysis.get('protein_existence_distribution', {})
            for existence, count in existence_dist.items():
                f.write(f"- {existence}: {count} 条\n")
        
        print(f"分析报告保存完成！")


def main():
    filter = NonAntimicrobialPeptideFilter()
    
    input_csv = "short_peptides_complete.csv"
    output_csv = "non_antimicrobial_peptides_filtered.csv"
    output_report = "non_antimicrobial_peptides_filtered_report.md"
    
    try:
        df = filter.load_data(input_csv)
        
        filtered_df = filter.filter_peptides(df, max_score=0.0)
        
        analysis = filter.analyze_filtered_data(filtered_df)
        
        filter.save_filtered_data(filtered_df, output_csv)
        filter.save_analysis_report(analysis, output_report)
        
        print("\n" + "=" * 60)
        print("筛选完成！")
        print("=" * 60)
        print(f"\n输出文件:")
        print(f"  1. {output_csv} - 筛选后的CSV数据")
        print(f"  2. {output_report} - 分析报告")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
