# Antifungal Peptide Data Processing

抗真菌肽数据处理工具包

## 目录结构

```
all_antifungal/
├── dataset/                    # 原始数据文件
│   ├── Antifungal_amps.csv     # DRAMP抗真菌肽数据
│   ├── antifungal.csv          # 抗真菌数据
│   ├── antifungal_peptides.csv # 抗真菌肽数据
│   ├── merged_data.csv          # 合并数据
│   └── peptides_merged.csv     # 肽类合并数据
│
├── mmseq/                       # 聚类分析相关
│   ├── extract_representative_sequences.py
│   ├── merged_antifungal_le60.fasta
│   ├── merged_antifungal_le60_all_seqs.fasta
│   ├── merged_antifungal_le60_cluster.tsv
│   ├── merged_antifungal_le60_rep_seq.fasta
│   └── representative_sequences.csv
│
├── merge_csv_improved.py       # 合并CSV文件
├── filter_2plus_improved.py    # 2+文件过滤
├── filter_le60_improved.py     # 短肽过滤(≤60 aa)
├── filter_2plus_correct.py    # 2+文件过滤(修正版)
├── standardize_and_filter_improved.py  # 标准化与过滤
├── csv_to_fasta.py             # CSV转FASTA
└── verify_sequence_sources.py # 验证序列来源
```

## 数据处理流程

### 步骤1: 合并CSV文件
```bash
python merge_csv_improved.py
```
- 合并5个原始CSV文件
- 统一列名（Sequence, Name等）
- 合并元数据，保留优先级高的来源信息

### 步骤2: 标准化与过滤
```bash
python standardize_and_filter_improved.py
```
- 序列大写转换
- 过滤非标准氨基酸序列
- 过滤仅在1个文件中出现的序列（2+文件过滤）
- 输出: `merged_antifungal_all_improved_standardized.csv`
- 输出: `merged_antifungal_2plus_improved_standardized.csv`

### 步骤3: 短肽过滤
```bash
python filter_le60_improved.py
```
- 过滤长度≤60 aa的序列
- 比较新旧数据集差异

### 步骤4: CSV转FASTA
```bash
python csv_to_fasta.py
```
- 将CSV转换为FASTA格式
- 用于后续聚类分析

### 步骤5: 聚类分析 (MMseqs2)
```bash
# 需要先安装MMseqs2
mmseqs easy-cluster input.fasta cluster_res tmp
python mmseq/extract_representative_sequences.py
```
- 使用MMseqs2进行序列聚类
- 提取代表性序列

### 步骤6: 验证序列来源
```bash
python verify_sequence_sources.py
```
- 验证最终数据集中的序列来源
- 统计各文件的贡献

## 输出文件说明

| 文件名 | 说明 |
|--------|------|
| `merged_antifungal_all_improved.csv` | 改进的合并数据集（所有序列） |
| `merged_antifungal_2plus_improved_standardized.csv` | 2+文件过滤+标准化后的数据集 |
| `merged_antifungal_le60_final.csv` | 最终短肽数据集(≤60 aa) |
| `merged_antifungal_le60.fasta` | FASTA格式的短肽数据 |
| `mmseq/merged_antifungal_le60_rep_seq.fasta` | 聚类代表性序列 |
| `mmseq/representative_sequences.csv` | 代表性序列CSV格式 |

## 依赖

- Python 3.x
- pandas
- MMseqs2 (用于聚类分析)

## 数据源

- DRAMP (Data Repository of Antimicrobial Peptides)
- 抗真菌肽数据库
- 已发表的抗真菌肽数据
