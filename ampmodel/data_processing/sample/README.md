# 抗菌肽序列分析样本数据目录

本目录包含用于抗菌肽(AMP)序列探索性分析和可视化的Python脚本及相关数据文件。

## 目录结构

```
sample/
├── figures/                          # 生成的图表文件
│   ├── fig1_length_distribution.*   # 序列长度分布
│   ├── fig2_amino_acid_properties.* # 氨基酸性质
│   ├── fig3_aa_frequency_difference.* # 氨基酸频率差异
│   ├── fig4_net_charge.*           # 净电荷分布
│   ├── fig5_hydrophobicity_distribution.* # 疏水性分布
│   ├── fig6_charge_distribution.*   # 电荷分布
│   ├── fig7_shannon_entropy.*       # Shannon熵
│   ├── fig8_n_terminal.*             # N端氨基酸
│   ├── fig9_c_terminal.*             # C端氨基酸
│   ├── fig10_hydrophobicity_vs_charge.* # 疏水性vs电荷
│   └── fig11_detailed_aa_categories.*  # 氨基酸详细分类
│
├── analyze_sequences.py              # 序列基本统计分析
├── compare_sequences.py              # 序列对比分析(抗真菌vs非抗菌)
├── detailed_sequence_analysis.py     # 详细序列分析
├── extract_id_sequence.py            # 提取ID和序列
├── generate_figures.py               # 顶刊级可视化(扩展版)
├── generate_comparison_figures.py    # 对比分析可视化
├── generate_publication_figures.py   # 顶刊级出版图表
├── visualize_analysis.py             # 可视化分析
│
├── id_sequence_extracted.csv        # 提取的ID-序列数据
├── representative_sequences.csv      # 代表性序列
├── representative_sequences_filtered_no_x.csv # 过滤后的序列
├── representative_sequences_id_seq.csv # ID-序列格式
├── sequence_stats.json               # 序列统计数据
└── comparison_stats.json             # 对比统计数据
```

## 脚本说明

### 数据处理脚本

| 脚本 | 功能描述 |
|------|---------|
| `extract_id_sequence.py` | 从CSV文件中提取`representative_id`和`sequence`列，保存为新文件 |
| `analyze_sequences.py` | 抗菌肽序列基本统计分析：长度分布、氨基酸组成、非标准氨基酸识别 |
| `compare_sequences.py` | 抗真菌序列与非抗菌序列的对比分析：长度统计、氨基酸性质分布(疏水性/极性/电荷) |
| `detailed_sequence_analysis.py` | 详细分析：长度分位数、氨基酸频率差异显著性检验(Fisher精确检验)、氨基酸分类统计 |

### 可视化脚本

| 脚本 | 功能描述 |
|------|---------|
| `visualize_analysis.py` | 基础可视化：长度分布直方图、氨基酸组成对比、性质分类柱状图 |
| `generate_figures.py` | 顶刊级可视化(扩展版)：Kyte-Doolittle疏水性计算、多种统计图表 |
| `generate_comparison_figures.py` | 对比分析可视化：抗真菌vs非抗菌序列的长度分布、氨基酸组成雷达图 |
| `generate_publication_figures.py` | 出版级图表生成：遵循Nature/Science期刊规范，使用Okabe-Ito颜色方案 |

### 图表输出格式

- **PDF**: 矢量格式，适合论文出版
- **PNG**: 300 DPI raster格式，适合演示和网页

## 数据文件说明

| 文件名 | 描述 |
|--------|------|
| `id_sequence_extracted.csv` | 提取的ID和序列对，包含`representative_id`和`sequence`两列 |
| `representative_sequences.csv` | 代表性抗菌肽序列原始数据 |
| `representative_sequences_filtered_no_x.csv` | 过滤掉含'X'氨基酸的序列 |
| `representative_sequences_id_seq.csv` | 统一格式的ID-序列文件 |
| `sequence_stats.json` | 序列基本统计信息(JSON格式) |
| `comparison_stats.json` | 对比分析统计数据 |

## 主要分析内容

1. **序列长度分析**: 最小/最大/平均/中位数长度，长度分布分箱
2. **氨基酸组成**: 20种标准氨基酸频率统计，非标准氨基酸检测
3. **氨基酸性质分类**:
   - 疏水性: A, V, I, L, M, F, W, P
   - 极性: S, T, N, Q, Y, C
   - 正电荷: K, R, H
   - 负电荷: D, E
4. **物理化学性质**: 净电荷、疏水性(Scale: Kyte-Doolittle)、Shannon熵
5. **末端分析**: N端和C端氨基酸分布
6. **统计检验**: t检验、Fisher精确检验

## 使用方法

```bash
# 基本序列分析
python analyze_sequences.py

# 序列对比分析
python compare_sequences.py

# 生成可视化图表
python generate_publication_figures.py
```

## 依赖库

- pandas
- numpy
- matplotlib
- seaborn
- scipy

## 图表规范

本项目生成的图表遵循科学出版标准：
- 字体: Arial/Helvetica (无衬线)
- 颜色: Okabe-Ito 颜色盲友好调色板
- 输出: PDF (矢量) + PNG (300 DPI)
- 避免Type3字体，使用TrueType字体
