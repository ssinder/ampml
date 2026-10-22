# UniProt Peptide Non-Antimicrobial Peptide Data Processing

本目录包含用于从 UniProt 数据库获取、处理和分析非抗菌肽数据的 Python 脚本和输出文件。

## 目录结构

```
uniprot_peptide/
├── 数据获取脚本
│   ├── select_non_antimicrobial.py    # 从 UniProt 选择非抗菌肽
│   ├── query_short_peptides.py     # 查询短肽数据
│   ├── query_go_terms.py          # 查询 GO 术语信息
│   └── query_short_peptides_complete.py  # 完整短肽查询
│
├── 数据过滤脚本
│   ├── filter_non_antimicrobial.py        # 过滤抗菌肽（非抗菌肽筛选）
│   ├── filter_non_antimicrobial_strict.py # 严格过滤（非抗菌肽）
│   ├── filter_complete_metadata.py      # 过滤完整元数据
│   └── filter_x_sequences.py          # 过滤含X的序列
│
├── 格式转换脚本
│   ├── csv_to_fasta.py       # CSV 转换为 FASTA 格式
│   ├── export_to_fasta.py    # 导出为 FASTA 格式
│   └── export_to_csv.py      # 导出为 CSV 格式
│
├── 数据分析脚本
│   ├── analyze_rep_seq_features.py        # 分析代表性序列特征
│   ├── analyze_rep_seq_features_no_x.py   # 分析代表性序列（不含X）
│   ├── analyze_mmseq_clustering.py          # 分析 MMSeq 聚类结果
│   └── analyze_filtered_clustering.py     # 分析过滤后聚类结果
│
├── mmseq/                    # MMSeq 聚类分析目录
│   ├── representative_sequences_filtered.csv    # 过滤后代表性序列
│   ├── representative_sequences_filtered_no_x.csv # 过滤后代表性序列（不含X）
│   ├── representative_sequence_analysis_filtered.txt  # 聚类分析报告
│   └── non_antimicrobial_peptides_high_quality_complete_filtered*  # 过滤后数据文件
│
└── 数据文件（CSV/FASTA/JSON）
    ├── non_antimicrobial_peptides_high_quality.csv       # 高质量非抗菌肽
    ├── non_antimicrobial_peptides_high_quality_complete.csv # 完整高质量非抗菌肽
    ├── non_antimicrobial_peptides_high_quality_complete_filtered.csv # 过滤后数据
    └── non_antimicrobial_peptides*.fasta                  # FASTA 格式文件
```

## 主要功能说明

### 1. 数据获取
通过 UniProt REST API 查询蛋白质数据，支持：
- 按关键词搜索
- 短肽过滤（长度 ≤ 100 aa）
- GO 术语提取
- 速率限制处理

### 2. 抗菌肽过滤
基于以下规则过滤掉抗菌肽：
- **抗菌相关 GO terms**: GO:0042742, GO:0031640, GO:0006952, GO:0090729 等
- **抗菌关键词**: antimicrobial, defensin, antibiotic, bacteriocin, toxin 等
- **分泌相关 GO terms**: GO:0005576 (细胞外区域), GO:0005615 (细胞外空间)

### 3. 数据质量控制
- 序列完整性检查
- 氨基酸有效性验证
- 去除含未知氨基酸 (X) 的序列
- 高质量序列筛选

### 4. 聚类分析
使用 MMSeq2 对序列进行聚类：
- 生成代表性序列
- 计算聚类统计信息
- 分析序列长度分布
- 氨基酸组成分析

## 使用示例

```python
# 筛选非抗菌肽
from filter_non_antimicrobial import NonAntimicrobialPeptideFilter
filter = NonAntimicrobialPeptideFilter()
df = filter.load_data("input.csv")
filtered_df = filter.filter_antimicrobial_peptides(df)

# CSV 转 FASTA
from csv_to_fasta import csv_to_fasta
csv_to_fasta("input.csv", "output.fasta")

# 序列特征分析
python analyze_rep_seq_features.py
```

## 输出文件说明

| 文件名 | 说明 |
|--------|------|
| non_antimicrobial_peptides_high_quality.csv | 高质量非抗菌肽数据 |
| non_antimicrobial_peptides_high_quality_complete_filtered.csv | 过滤后完整数据 |
| non_antimicrobial_peptides_high_quality_complete_filtered.fasta | FASTA 格式数据 |
| representative_sequences_filtered.csv | 聚类代表性序列 |
| short_peptide_results.json | 短肽查询结果 |
| go_terms_analysis.json | GO 术语分析结果 |

## 依赖库

- pandas
- numpy
- requests

## 注意事项

- 所有 UniProt API 查询都有速率限制 (默认 1 秒/请求)
- 生成的 FASTA 文件使用 `|` 分隔符作为 header
- 聚类分析需要先安装 MMSeq2 工具