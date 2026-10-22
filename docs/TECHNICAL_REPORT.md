# AMP-ML 抗菌肽机器学习分类系统 技术报告

| 项    | 内容                                                         |
| ---- | ---------------------------------------------------------- |
| 项目名称 | AMP-ML（Antibacterial Peptide Machine Learning）             |
| 系统范围 | 上游数据构建（`ampmodel/`）+ 下游建模审计（`src/`）双子系统                    |
| 主推模型 | Gradient Boosting（Acc 0.8646 / F1 0.8706 / ROC-AUC 0.9432） |
| 验收状态 | 五层审计全部通过，30/30 项数值复算一致，退出码 0                               |
| 报告日期 | 2026-10-20                                                 |
| 数值来源 | `results/` 与 `data/` 下权威 CSV/JSON，代码引用均给出行号                |
| 配套文档 | 复现步骤见 `docs/REPRODUCIBILITY.md`；验收执行手册见 `docs/THIRD_PARTY_VERIFICATION.md`；审计结论见 `results/VERIFICATION_SUMMARY.md` |

---

## 一、项目组成

### 1.1 系统定位与总体目标

AMP-ML 是一个面向**肽段序列**的监督式二分类系统。输入为一条氨基酸序列经特征提取后的 4921 维数值向量，输出为该肽段是否具有抗菌活性（L1=抗菌肽，L0=非抗菌肽）的判定及置信概率。

系统由**两个子系统**构成：

| 子系统            | 位置          | 职责                        | 复现承诺               |
| -------------- | ----------- | ------------------------- | ------------------ |
| **上游数据构建子系统**  | `ampmodel/` | 序列获取 → 筛选去冗余 → 4921 维特征提取 | 特征提取层可复现；序列构建层不可复现 |
| **下游建模与验收子系统** | `src/`      | 预处理 → 特征选择 → 建模 → 解释 → 审计 | 完全可复现              |

系统解决的核心工程问题是：**在 2557 条样本、4921 维高维特征的条件下，构建一条可被第三方逐位复现、并自带审计证据的建模链路。** 因此系统不以单点精度为唯一目标，而将"可复现性"和"可验收性"作为一等非功能需求内建到架构中。

### 1.2 软硬件与运行环境

**硬件平台（实测）**

| 项   | 配置                               |
| --- | -------------------------------- |
| CPU | Apple M4 Pro，12 核                |
| 内存  | 24 GB                            |
| 架构  | arm64（macOS 15.7.5，Build 24G624） |

**软件环境（`requirements.txt` 全量锁定，共 38 个包）**

| 层次   | 组件                   | 版本              | 作用               |
| ---- | -------------------- | --------------- | ---------------- |
| 运行时  | CPython              | 3.13.9          | 解释器基线（自检要求 ≥3.9） |
| 数据处理 | pandas / numpy       | 3.0.2 / 2.4.4   | 表格与数值计算          |
| 机器学习 | scikit-learn         | 1.8.0           | 建模、CV、特征选择、预处理   |
| 集成学习 | XGBoost              | 3.2.0           | 梯度提升树            |
| 可解释性 | SHAP                 | 0.51.0          | 归因解释             |
| 统计   | scipy                | 1.17.1          | 分布检验、ROC/PR 曲线   |
| 可视化  | matplotlib / seaborn | 3.10.8 / 0.13.2 | 学术图表             |
| 上游特征 | modlamp / biopython  | 4.3.2 / 1.87    | 上游 12 维理化描述符     |
| 并行   | joblib               | 1.5.3           | `n_jobs=-1` 并行调度 |

> 跨大版本重训会得到不同数值（例如 sklearn 1.7 与 1.8），因此版本锁定被列为复现必要条件之一。

**网络与外部依赖**

下游主链路设计为**完全离线**：`run_all.sh` 全流程不访问网络，模型推理为本地 pickle 加载。仅上游序列获取阶段依赖 UniProt REST、DRAMP/dbAMP/APD/CAMPR3/DBAASP 的本地导出文件，以及 MMseqs2 去冗余——该部分不纳入本报告的复现承诺范围。

### 1.3 系统总体结构

系统采用**五层管道式架构**，编号即执行顺序，每层通过磁盘文件契约解耦，可独立重跑：

```
┌─────────────────────────────────────────────────────────────────┐
│  L5 验收审计层   99_verify_test_metrics.py                       │
│      只读复算 + 五层核对 → verification_report.json / SUMMARY.md │
├─────────────────────────────────────────────────────────────────┤
│  L4 解释与展示层  05_shap_analysis.py / 04b_generate_figures…   │
│      SHAP 归因 / 学术图表          → results/shap/ · figures/   │
├─────────────────────────────────────────────────────────────────┤
│  L3 建模层      03_model_training_cv.py / 03b_generate_cv…     │
│      6 模型 × (RandomizedSearch + 嵌套CV + 5折CV) → results/   │
├─────────────────────────────────────────────────────────────────┤
│  L2 特征层      02_feature_selection.py                        │
│      5 方法竞赛 4921 → 300                    → data/*_selected │
├─────────────────────────────────────────────────────────────────┤
│  L1 数据层      01_data_preparation.py                         │
│      加载/校验/填充/标准化/分层划分(70/15/15)   → data/*.csv     │
└─────────────────────────────────────────────────────────────────┘
       ▲ 横向支撑：config.py（唯一配置源）· run_all.sh（编排）· 00 环境自检
──────────────────────────────────────────────────────────────────
       ▼ 上游数据构建子系统（部分可复现，见 1.7）
   ampmodel/  序列获取 → 筛选去冗余 → 特征提取 → rawdata/*.csv (2557×4921)
──────────────────────────────────────────────────────────────────
```

**横向支撑组件**贯穿各层：

- `src/config.py`——全局配置唯一事实来源，持有路径、随机种子、划分比例、折数、目标特征数、原始数据 MD5；
- `run_all.sh`——8 阶段流水线编排器，支持 `--list/--from/--only/--clean/--skip-check`，失败即停（`set -euo pipefail`），逐阶段计时并落盘日志；
- `src/00_check_environment.py`——第三方验收第一道关卡，7 类前置校验。

### 1.4 模块划分与职责

**下游建模子系统（`src/`）**

| 编号  | 模块      | 输入                | 输出                                                                              | 职责                                          |
| --- | ------- | ----------------- | ------------------------------------------------------------------------------- | ------------------------------------------- |
| 00  | 环境自检    | 环境、rawdata        | 退出码 + 校验清单                                                                      | Python 版本、11 包版本、原始文件 MD5、必需输入/产物齐全性、产物内容抽检 |
| 01  | 数据预处理   | rawdata CSV       | `train/val/test_data.csv`、`sample_id_mapping.csv`、`preprocessing_metadata.json` | 标签提取、缺失/常量/异常检测、中位数填充、标准化、分层划分              |
| 02  | 特征选择    | train/val/test    | `*_data_selected.csv`、`selected_features.csv`、`feature_selection_metadata.json` | 5 种方法同台 CV 竞赛，自动选优，4921→300                 |
| 03  | 模型训练    | selected CSV      | 6 个 `.pkl`、`cv_results_summary.csv`、逐折明细、ROC/PR 曲线点、混淆矩阵                        | 随机搜索超参 + 嵌套 CV + 5 折 CV + 测试集曲线数据           |
| 03b | CV 曲线数据 | 模型 + train        | CV 折外预测曲线点                                                                      | 补齐 CV 曲线数据                                  |
| 04b | 图表生成    | 6 个模型 + test      | `fig1`~`fig4`（300 dpi PNG）                                                      | ROC、混淆矩阵、PR、模型对比四联图                         |
| 05  | SHAP 分析 | 4 个树/线性模型         | `results/shap/`（含 `shap_values/` 明细）                                            | 归因计算、重要性排名、模型间相关性、Top20 交集                  |
| 06  | 集成预测    | voting pkl + 全量特征 | `results/predictions.csv`                                                       | 对全量样本批量推理并输出概率                              |
| 99  | 验收审计    | 6 个 pkl + test    | `verification_report.json`、`VERIFICATION_SUMMARY.md`                            | 五层递进审计，0/1 退出码                              |

**上游数据构建子系统（`ampmodel/`）**

| 目录                                 | 职责                                                             | 关键产物                                                                                          |
| ---------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `data/`                            | 7 个原始数据库的导出/抓取结果（APD6、dbAMP、DRAMP、CAMPR3、DBAASP 等，含抗菌与抗真菌两个方向） | 各库 CSV/FASTA 原件                                                                               |
| `data_processing/all_antifungal/`  | **L1（抗菌肽）链路**：5 库合并 → 标准化 → 2+ 库交叉验证 → 长度过滤 → MMseqs2 去冗余      | `merged_antifungal_le60_final.csv`(4758) → `representative_sequences.csv`(2255)               |
| `data_processing/uniprot_peptide/` | **L0（非抗菌肽）链路**：UniProt 查询 → 启发式否定筛选 → 质量分级 → MMseqs2 去冗余       | `non_antimicrobial_peptides_*.csv`(4119) → `representative_sequences_filtered_no_x.csv`(1929) |
| `data_processing/sample/`          | 两条链的**装配点**，输出统一的 (ID, Sequence) 两列表                           | `id_sequence_extracted.csv`(1929)、`representative_sequences_id_seq.csv`(2255)                 |
| `feature_extraction/features/`     | 特征提取层：调用 iFeature / PseKRAAC / modlAMP，合并为 4921 维              | `all_features_combined_standard.csv`                                                          |
| `model_training/`                  | 早期特征选择与建模实验，功能上已被 `src/` 全面取代（详见 1.8）                          | 历史实验产物                                                                                        |

### 1.5 模块间关系与接口契约

下游模块间**无直接函数调用**，一律通过文件系统交接，契约由三条规则固化：

1. **列序契约**：所有数据 CSV 的标签列必须是最后一列，下游统一以 `iloc[:, :-1]` 取特征、`iloc[:, -1]` 取标签（见 `src/02_feature_selection.py:56-60`、`src/03_model_training_cv.py:70-76`）；
2. **随机种子契约**：`RANDOM_STATE = 42` 是唯一来源（`src/config.py:33`），任何脚本不得自行定义；
3. **路径契约**：全部路径由 `config.py` 的 `ROOT` 推导（`src/config.py:16`），禁止硬编码绝对路径，脚本可从任意工作目录执行。

> 样本 ID 未内嵌于主 CSV（否则破坏列序契约），而是另存于 `data/sample_id_mapping.csv`，含 `split / row_index / ID / label` 四列，实现行级回溯（`src/01_data_preparation.py:440-446`）。

### 1.6 基本数据结构

**数据集结构**

| 项       | 值                                             |
| ------- | --------------------------------------------- |
| 样本总数    | 2557（L0=1231 / L1=1326）                       |
| 原始特征维度  | 4921                                          |
| 入选特征维度  | 300（压缩 93.9%）                                 |
| 划分      | train 1789 / val 384 / test 384（分层抽样，seed=42） |
| 训练集类别分布 | L0=861 / L1=928                               |

**4921 维特征构成（按表头前缀实测归类，差额 0）**

| 来源       | 维度   | 明细                                                                                                                                              |
| -------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| iFeature | 4484 | CKSAAP 2400 / DPC 400 / DDE 400 / KSCTriad 343 / CTriad 343 / CTDD 195 / CKSAAGP 150 / GTPC 125 / CTDT 39 / CTDC 39 / GDPC 25 / AAC 20 / GAAC 5 |
| PseKRAAC | 425  | 17 种氨基酸类型 × 25 组成型 k-mer                                                                                                                        |
| modlamp  | 12   | Length / Isoelectric_Point / Hydrophobic_Ratio / Eisenberg_Moment / KyteDoolittle_Moment / AutoCorr 0–6                                         |

> **`ID.1` 列说明**：`ID.1` 是 DPC 描述符中 Ile-Asp 二肽组成特征（Isoleucine + Aspartic acid 的二字母缩写），因与首列 `ID` 列重名而在下游被 pandas 重命名，并非冗余列。判定依据：上游 `iFeature_DPC.tsv` 第 144 个表头即为 `ID`；且 400 个二肽特征逐行求和精确等于 1.0（实测均值 1.000000000000，标准差 9.65e-16），满足组成型特征的归一化守恒。**该列参与 4921 维口径，不予剔除。**

**核心数据文件**

| 文件                                           | 规模                 | 用途              |
| -------------------------------------------- | ------------------ | --------------- |
| `rawdata/all_features_combined_standard.csv` | 2557 × 4922，78 MB  | 上游产物，含 ID 列     |
| `data/train_data.csv`                        | 1789 × 4922，170 MB | 标准化后全特征训练集      |
| `data/train_data_selected.csv`               | 1789 × 301，10.6 MB | 300 维建模训练集      |
| `data/sample_id_mapping.csv`                 | 2557 行             | 行级样本溯源          |
| `results/models/*.pkl`                       | 17 KB – 7.0 MB     | 6 个序列化 Pipeline |

> **`data/features_processed.csv`**：由 `src/01_data_preparation.py:416-419` 生成的 2557×4923 全量未筛选特征文件，作为 `06` 脚本的特征模板。当前 `data/` 目录下的产物为 2026-04-21 存档，该文件未随之一并留存，需重跑 `01` 再生（详见 2.4）。

**指标数据结构**：`cv_results_summary.csv` 每模型 20 列（8 项指标 × 均值/标准差 + 嵌套 CV 分数）；`verification_report.json` 为五层嵌套结构（`data_audit` / `model_files` / `models` / `summary`），人读版由脚本自动渲染为 Markdown（`src/99_verify_test_metrics.py:267-334`）。

### 1.7 上游数据构建链路

上游是独立于 `src/` 的数据供应链，产出 `rawdata/all_features_combined_standard.csv`（MD5 `7ce771dc55f1cf66d0dd28f8609c0bcb`，与 `src/config.py:29` 登记值一致）。

**L1 抗菌肽链路**（`ampmodel/data_processing/all_antifungal/`）

| 步 | 处理                                                                                  | 数量                    |
| - | ----------------------------------------------------------------------------------- | --------------------- |
| 1 | 5 库合并（dbAMP 5708 / DBAASP 6892 / DRAMP 1836 / APD 1813 / CAMPR3 562），统一列名、按优先级打分选主行 | 16,811 → **7,920**    |
| 2 | 序列大写化 + 过滤非标准氨基酸 + 要求至少在 2 个数据库中出现                                                  | **6,921** → **5,306** |
| 3 | 长度 ≤60 过滤                                                                           | **4,758**             |
| 4 | MMseqs2 聚类去冗余取代表序列                                                                  | **2,255**（压缩 47.4%）   |
| 5 | 标签装配 + 过滤（标准 AA、长度 ≥10）                                                             | **1,326**             |

**L0 非抗菌肽链路**（`ampmodel/data_processing/uniprot_peptide/`）

| 步 | 处理                                                   | 数量                |
| - | ---------------------------------------------------- | ----------------- |
| 1 | UniProt REST 查询 `reviewed:true AND length:[1 TO 60]` | **18,171**        |
| 2 | 启发式否定筛选（GO term + 关键词加权评分，保留得分 ≤0）                   | **14,006**        |
| 3 | 保留 `protein_existence` 含 "Evidence at protein level" | **4,942**         |
| 4 | 要求 `keywords` 与 `go_terms` 均非空                       | **4,281**         |
| 5 | 按 LLM 辅助生成的蛋白名黑名单（`llm_clean.md`）剔除                  | **4,119**         |
| 6 | MMseqs2 聚类去冗余 → 滤 X 氨基酸                              | 2,225 → **1,929** |
| 7 | 标签装配 + 过滤                                            | **1,231**         |

> **标签归属提示**：目录名与标签相反——`uniprot_peptide` 产出 **L0**，`all_antifungal` 产出 **L1**。证据见 `ampmodel/feature_extraction/features/generate_labeled_sequences.py:15-16`（读 `sample/` 两个中转文件）、`:41-46`（L0 前缀赋值）、`:49-54`（L1 前缀赋值）。

**特征提取层**（`ampmodel/feature_extraction/features/extract_features.py`）

| 工具       | 调用方式                                                                             | 产出维度         |
| -------- | -------------------------------------------------------------------------------- | ------------ |
| iFeature | `subprocess` 调 `docs/github/iFeature-master/iFeature.py`，18 种 type               | 4,484        |
| PseKRAAC | `subprocess` 调 `iFeaturePseKRAAC.py`，`--subtype g-gap --ktuple 2 --gap_lambda 2` | 425（落盘 17 种） |
| modlAMP  | Python API `PeptideDescriptor` / `GlobalDescriptor`                              | 12           |

> **上游提取缺口（实测）**：iFeature 的 **5 个描述符计算失败**（APAAC / Geary / Moran / NMBroto / PAAC，产物文件内容为 `Descriptor calculation failed.`），PseKRAAC 的 type5 与 type13 无产物。**4921 维正是扣除这些失败项后的实际维度**——这解释了维度对账为何能差额为 0，也意味着"4921 维"是失败项计入后的口径，而非 iFeature 全部 18 种类型的理论维度。

`rename_features.py` **只改表头、不做任何数值变换**：剥离 `iFeature_` / `modlamp_` 前缀，`PseKRAAC_` 前缀保留，列数与数值逐字节不变。文件名中的 `_standard` 指**标准化命名**，非数值标准化——真正的数值标准化产物是上游另一个文件 `features_standardized.csv`（1,089 列），下游未使用。

### 1.8 已废弃的上游建模目录

`ampmodel/model_training/` 是上游项目遗留的探索性实验目录，功能上已被 `src/` 全面取代：

| 上游遗留                                                                  | 下游取代者                                      |
| --------------------------------------------------------------------- | ------------------------------------------ |
| `test/smalltest/prepare_data.py`（标准化+划分）                              | `src/01_data_preparation.py`               |
| `test/{statistical,randomforest,elasticnet,rfe}_feature_selection.py` | `src/02_feature_selection.py`（5 法竞赛）       |
| `smalltest/model_training.py`（200 样本小规模实验）                            | `src/03_model_training_cv.py`（6 模型 5 折 CV） |
| `smalltest/shap_analysis.py`                                          | `src/05_shap_analysis.py`                  |
| `smalltest/publication_plots.py`                                      | `src/04b_generate_figures_optimized.py`    |

**关键差异**：上游 `model_training/` 基于 **1,098 维**上游预处理特征做选择，下游 `src/` 基于 **4,921 维**原始特征做选择——两条路的特征空间不同，**上游 `smalltest/` 的任何数值都不可被下游继承**。该目录下全部脚本硬编码 `D:/trae/paper/...` Windows 路径，亦不遵守下游的路径契约。

---

## 二、实现的主要功能、性能及其他非功能特性

### 2.1 主要功能

1. **数据治理**：加载校验、标签解析、缺失值检测、中位数填充、常量特征识别、IQR 异常值检测（阈值 3.0）、Z-score 标准化、三路分层划分。
2. **特征降维**：5 种特征选择方法在统一评估协议（同 LR 分类器 + 5 折 CV + ROC-AUC）下竞赛，自动择优。
3. **模型训练与调优**：5 个基模型 + 1 个软投票集成，各自随机搜索超参（n_iter 20~30）、嵌套交叉验证、完整 5 折 CV。
4. **模型持久化**：6 个模型以 pickle 序列化为完整 Pipeline（含 scaler），可独立加载推理。
5. **可解释性**：SHAP 归因计算，全局重要性排名、模型间重要性相关性、Top20 交集/独有分析。
6. **学术图表**：300 dpi 四联图（ROC / 混淆矩阵 / PR / 模型对比），数据全部来自已落盘中间 CSV，支持不重训改图。
7. **批量预测**：集成模型对全量样本推理，输出逐样本概率。
8. **验收审计**：五层递进核对，输出机读 + 人读双份报告，并给出明确的不通过退出码。
9. **一键复现**：单条 `bash run_all.sh` 从原始数据复现全部结果，逐阶段日志留痕。

### 2.2 性能指标

**独立测试集（验收口径，n=384，2026-10-19 复算）**

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC | 漏检率 | 达标(≥80%) |
|------|----------|-----------|--------|-----|---------|--------|-----------|
| **Gradient Boosting**（主推） | **0.8646** | 0.8621 | 0.8794 | **0.8706** | **0.9432** | **12.06%** | ✅ |
| Voting Ensemble | 0.8646 | 0.8657 | 0.8744 | 0.8700 | 0.9369 | 12.56% | ✅ |
| Random Forest | 0.8464 | 0.8646 | 0.8342 | 0.8491 | 0.9322 | 16.58% | ✅ |
| SVM | 0.8438 | 0.8564 | 0.8392 | 0.8477 | 0.9132 | 16.08% | ✅ |
| XGBoost | 0.8438 | 0.8458 | 0.8543 | 0.8500 | 0.9354 | 14.57% | ✅ |
| Logistic Regression | 0.8203 | 0.8186 | 0.8392 | 0.8288 | 0.9049 | 16.08% | ✅ |

主推模型混淆矩阵：TN=157 / FP=28 / FN=24 / TP=175。漏检率 FN/(FN+TP)=12.06%（注意区别于 FN 占全测试集的 6.25%）。

**训练集 5 折交叉验证（辅助参考，不作验收依据）**

| 模型 | Accuracy | F1 | ROC-AUC | PR-AUC | 嵌套 CV |
|------|----------|-----|---------|--------|---------|
| Logistic Regression | 0.9385±0.011 | 0.940 | 0.9867 | 0.988 | 0.9872 |
| SVM | 0.9245±0.018 | 0.925 | 0.9784 | 0.981 | 0.9792 |
| Voting Ensemble | 0.9078±0.012 | 0.909 | 0.9733 | 0.977 | — |
| XGBoost | 0.8720±0.012 | 0.874 | 0.9379 | 0.950 | 0.9378 |
| Gradient Boosting | 0.8714±0.013 | 0.874 | 0.9375 | 0.950 | 0.9394 |
| Random Forest | 0.8603±0.009 | 0.861 | 0.9341 | 0.947 | 0.9320 |

> **重要口径差异**：LR 在 CV 上最优（0.9385）却在独立测试集上最差（0.8203），是本项目最典型的过拟合证据。项目据此在 `results/README.md` 中明文规定"两套口径不可混用，CV 指标不作验收依据"。

**特征选择方法竞赛结果**

| 方法 | CV ROC-AUC | 特征数 | 最优参数 | 入选 |
|------|------------|--------|----------|------|
| **L1_Logistic** | **0.8899 ± 0.0153** | 300 | C=0.1 | ✅ 采纳 |
| Mutual_Info | 0.8896 ± 0.0099 | 200 | k=200 | |
| RF_Importance | 0.8861 ± 0.0170 | 300 | n_estimators=200 | |
| SelectKBest_Ftest | 0.8827 ± 0.0066 | 200 | k=200 | |
| SelectPercentile | 0.8795 ± 0.0109 | 246 | percentile=5 | |

**SHAP 识别的重要特征（按模型实测，见 3.7 的引用范围说明）**

| 模型 | Top 1 | Top 2 | Top 3 |
|------|-------|-------|-------|
| Logistic Regression | `M` (0.4569) | `Isoelectric_Point` (0.3288) | `KyteDoolittle_Moment` (0.2972) |
| XGBoost | `KyteDoolittle_Moment` (0.5304) | `CTDC_hydrophobicity_ARGP820101.G1` (0.4973) | `CTDD_charge.3.residue100` (0.3706) |
| Random Forest | `CTDC_hydrophobicity_ARGP820101.G1` (0.0283) | `Isoelectric_Point` (0.0281) | `CTDT_charge.Tr1221` (0.0271) |

其中 `Eisenberg_Moment`（LR 排名第 4）、`KyteDoolittle_Moment`（LR 第 3 / XGBoost 第 1）是经典疏水指数加权矩，与抗菌肽膜插入机制直接相关，为模型决策提供了领域可解释的支撑。

> 上表数据源：`results/shap/shap_summary.json`，按各模型 `mean_abs_shap` 降序取前三。完整排名见 `results/shap/feature_importance_ranking.csv`。

**运行开销（Apple M4 Pro 实测）**

| 阶段 | 耗时 |
|------|------|
| 02 特征选择（含 5 方法竞赛 + 多参数网格） | 403.9 s（占端到端约 64%，最慢环节） |
| 全流程 8 阶段 | ≈ 15 min |

模型文件体积：LR 17 KB、XGB 464 KB、GB 903 KB、SVM 2.3 MB、RF 6.4 MB、Voting 7.0 MB——单模型推理常驻内存开销可忽略。

### 2.3 非功能特性

**可复现性**

| 环节 | 状态 | 证据 |
|------|------|------|
| 下游 ML 全链路 | 完全可复现 | 48 项指标重跑比对：42 项差异为 0，AUC 类 6 项漂移 ≤ 3.7e-06 |
| 上游特征提取（4921 维） | 完全可复现 | modlamp 12 维逐位一致（maxdiff ≤ 4.4e-16） |
| 上游序列构建（L0/L1） | 不可复现 | MMseqs2 参数缺失 + 部分中间产物生成脚本缺失 |

复现的三个必要条件已在工程上落实：随机种子统一（config.py）、环境锁定（requirements.txt，38 包全量 pin）、划分参数不变（70/15/15，seed=42）。

**复现强度的实测结论**：隔离目录下端到端重跑 8 阶段，48 项指标与交付基线的差异全部为 0.00e+00（逐模型 Accuracy 差异 ≤3.33e-11，属浮点末位噪声），审计判定 30/30 项匹配。但模型文件 **MD5 并非全部一致**——RF / XGBoost / GB 三个一致，LR / SVM / Voting Ensemble 三个不同，原因是 pickle 内嵌 `n_jobs=-1` 等并行参数与运行时对象状态，字节级不可复现，但预测结果完全一致。

> 由此得出一条使用约束：**L3 模型指纹只能用于校验「交付文件未被篡改」，不能用于校验「重训结果是否等价」。** 二者需分开表述，不可混为一谈。

**可审计性**：审计报告明确声明自身边界——仅验证"已交付模型在已交付测试集上的指标可复算一致"，不验证训练过程是否使用测试集信息、不验证上游数据库构建过程、不验证对全新肽段的泛化能力（`src/99_verify_test_metrics.py:326-332`）。这种主动划定审计范围的做法本身是非功能设计的组成部分。

**可维护性**：脚本编号即执行顺序；每个脚本内函数职责单一（如 `detect_outliers_iqr` / `manual_metrics` / `md5_of`）；图表与中间数据解耦，支持不重训改图；所有中间产物落盘为 CSV/JSON/NPY，可追溯每一张图的数据来源。

**可靠性**：`run_all.sh` 使用 `set -euo pipefail`，任一阶段失败立即停止并保留完整日志；`00_check_environment.py` 以退出码阻断环境不匹配的执行；原始数据用 MD5 校验防篡改。

**安全性**：模型以 pickle 加载，仅加载本项目自身产出的文件，不引入外部不可信模型。

**结论适用边界（客观事实声明）**
- MMseqs2 聚类参数无记录 → 代表序列无法重生成；
- 6 条序列同时带 L0/L1 标签（上游序列级交集 7 条，经标准 AA + 长度 ≥10 过滤后为 6 条，与 `sample_id_mapping.csv` 的 ID 前缀校验口径一致）；
- L0 实为全蛋白 60 aa 片段，与 L1 天然抗菌肽分布不可比——这是本数据集最大的方法学局限，会压低可达精度上限。

### 2.4 当前可用性状态

需要区分"交付时可用"与"当前仓库状态"：

| 模块 | 状态 | 说明 |
|------|------|------|
| 00 / 01 / 02 / 03 / 03b / 04b / 05 / 99 | ✅ 可运行 | `bash run_all.sh` 全流程实测通过 |
| **06 集成预测** | ⚠ **需补一步产物** | 依赖 `data/features_processed.csv`，该文件未随 2026-04-21 存档一并留存。执行 `src/01_data_preparation.py` 即可再生（生成逻辑见 `src/01_data_preparation.py:416-419`） |
| `results/predictions.csv` | ⚠ 历史存档 | 时间戳 2026-04-16，内容有效但非最新一轮产出 |

> 复现路径：`bash run_all.sh` 覆盖 8 个阶段（含 01），执行后 `data/features_processed.csv` 自动生成，`06` 即可正常执行。`run_all.sh` 的阶段表当前**不含 `06`**，一键复现流程不会自动执行该步，需单独调用 `src/06_voting_ensemble_prediction.py`。

---

## 三、采用和解决的关键技术或要点

### 3.1 高维肽序列特征工程体系

**要点**：模型上限由特征体系决定。项目采用"三源互补"策略，避免单一描述符的偏置：

- **iFeature（4484 维）**：覆盖组成型（CKSAAP）、理化性质（DPC/DDE/GTPC）、序列关联（KSCTriad/CTriad）、CTD 类理化分布等 14 个家族，是主力信号源；
- **PseKRAAC（425 维）**：拟态 k-mer 氨基酸组成（pseudo-k-mer reduced amino acid composition），在极低序列相似性下仍能捕捉组成偏好，对发现远缘同源新肽特别有效；
- **modlamp（12 维）**：理化描述符，提供跨特征家族的"语义锚点"——Eisenberg 矩、疏水比、等电点等，是 SHAP 排名中少数进入 Top 5 的全局特征，验证了低维高信息量特征的不可替代性。

**维度对账**：4484 + 425 + 12 = 4921，与文件实际列数差额为 0。该校验通过逐列前缀归类 + 上游产物回溯完成，可复现核对方法见 1.6。

### 3.2 防数据泄漏的预处理时序

**技术要点**：预处理顺序错误是机器学习项目最隐蔽的缺陷来源——若在全量数据上 fit 标准化器，测试集统计量会泄漏进训练，导致指标虚高。

项目采用**先划分、后拟合**的严格时序：

```
① 加载 2557×4921 → ② 划分 70/15/15（分层，seed=42）→ ③ 仅在 train 上拟合 Imputer 与 Scaler
→ ④ 用 train 的统计量 transform val/test → ⑤ 异常值检测也只在 train 上做
```

代码层面通过 `imputer.fit_transform(X_train)` + `imputer.transform(X_val/X_test)`（而非对三者分别 fit）强制保证这一约束（`src/01_data_preparation.py:194-196`）；`src/03_model_training_cv.py:86-97` 中模型再次以 `Pipeline(scaler → classifier)` 封装，形成第二重防护。

另一处易错点被显式处理：数据划分使用 DataFrame 按索引切分而非直接对 numpy 数组切分（`src/01_data_preparation.py:334-346`），从而完整保留原始行号，使 `sample_id_mapping.csv` 能把每一行预测回溯到具体肽序列 ID。

**一处口径提示**：模型 Pipeline 内的 `StandardScaler` 是在**已标准化过的**特征上再次标准化。由于 StandardScaler 是线性变换且对已零均值单位方差的输入是幂等操作，此处不引入偏差，但属冗余计算。

### 3.3 多方法同台竞赛的特征选择机制

**技术要点**：特征选择方法的选择本身构成超参数，若靠人工经验决定会引入隐性偏差。项目将其自动化：

1. **统一评估协议**：5 种方法（L1-Logistic、RF 重要性、互信息、ANOVA F-test、SelectPercentile）全部封装进 `Pipeline(selector → LR分类器)`，用同一个 5 折 StratifiedKFold + ROC-AUC 评分，杜绝因评估器不同导致的口径不可比（`src/02_feature_selection.py:76-79`）；
2. **方法内调参**：每种方法再搜索自己的超参（L1 的 C∈{0.1,0.5,1.0}、MI 的 k∈{200,250,300}、RF 的 n_estimators∈{100,200}、百分位的 p∈{5,10,15,20}）；
3. **流水线化防泄漏**：selector 被置于 Pipeline 内，CV 每折只在折内训练集上 fit，从机制上杜绝"先用全量数据筛特征再交叉验证"的信息泄漏；
4. **缺口补偿**：若 L1 选出特征不足 300，按系数绝对值降序补齐，保证输出维度稳定（`src/02_feature_selection.py:168-185`）。

最终 L1_Logistic 以 0.8899 胜出，维度 4921→300（压缩 93.9%）。这一压缩比在几乎不损失 CV 分数的前提下，显著降低了训练开销与过拟合风险。

### 3.4 嵌套交叉验证与超参数搜索的分离

**技术要点**：在同一份数据上既做超参搜索又做性能评估，会使报告的分数乐观偏置（"选择偏倚"）。项目用两个层次隔离这一偏差：

- **内层**：`RandomizedSearchCV`（n_iter 20~30，5 折 CV，scoring=roc_auc）搜索超参，`n_jobs=-1` 并行；
- **外层**：嵌套 CV（外层 5 折 seed=43，内层 3 折 GridSearch）评估"调参流程本身"的泛化能力（`src/03_model_training_cv.py:161-223`）；
- **最终**：取 `best_estimator_` 再做一轮完整 5 折 CV（seed=42），得到交付指标。

嵌套 CV 的分数（GB 0.9394）与同模型常规 CV（0.9375）接近，说明超参搜索未造成严重乐观偏置——这是一个值得记录的正面验证结果。经专项核查确认：内层 `param_grid` 是固定网格，不含外层已搜出的具体超参数值，内外层切分由 `CV_SEED=42` / `CV_OUTER_SEED=43` 隔离，实现上无泄漏。

需要说明的两点口径细节：

- `cv_results_summary.csv` 中的 `roc_auc_mean` 来自**全训练集选参后的 pipeline**，带轻微乐观偏置；无偏口径应参考 `nested_cv_score` 或独立测试集。该字段目前未参与选型、图表与 README，属记录性指标；
- **Voting Ensemble 无嵌套 CV**（`train_voting_ensemble_with_cv` 未调用 `nested_cv_evaluate`，CSV 中该列留空，`src/03_model_training_cv.py:375-376`）。

`Voting Ensemble` 采用软投票（soft voting）聚合 LR + RF + XGBoost 三者概率输出，理由是三者学习机制差异大（线性 / Bagging / Boosting），误差部分抵消。SVM 与 GB **不参与集成**（`src/03_model_training_cv.py:331` 显式过滤）。

**基学习器超参继承机制**：`src/03_model_training_cv.py:334` 对各基分类器调用 `clone()` 后传入 `VotingClassifier`。`clone()` 复制估计器**已设定的参数值**（源自 RandomizedSearchCV 的 `best_estimator_`），不重置为类默认值，因此超参搜索结果被正确继承。实测交付模型 `voting_ensemble_cv_model.pkl` 的基学习器参数与各自单独模型完全一致：LR `C=0.07887658410143283`；RF `n_estimators=300 / min_samples_split=10 / max_features='sqrt' / max_depth=None`；XGBoost `n_estimators=300 / max_depth=5 / learning_rate=0.1 / subsample=0.8`。集成采用 `voting='soft'`、`weights=None`（等权）。

### 3.5 单一事实来源与可复现性工程

**技术要点**：复现失败的常见原因不是算法，而是配置漂移——某处 `random_state=0`、某处硬编码路径、某处 `n_jobs` 影响线程数导致浮点求和顺序变化。

项目用制度化手段消除这类风险：

1. **`config.py` 单一事实源**：随机种子、路径、划分比例、CV 折数、目标特征数、原始数据 MD5 全部集中；所有脚本从此处 import，业务脚本中不存在自定的随机种子；
2. **嵌套 CV 种子隔离**：`CV_OUTER_SEED = RANDOM_STATE + 1`（`src/config.py:46`），避免内外层切分重合导致的评估偏乐观；
3. **依赖全量 pin**：`requirements.txt` 由 `pip freeze` 生成并加注释说明敏感项（如 modlamp 4.3.2 不可升级，否则 12 维描述符漂移）；
4. **原始数据指纹**：`RAW_FEATURES_MD5` 常量 + 环境自检逐字节校验，防止上游文件被静默修改。

**已量化结论**：重跑后 48 项指标中 42 项差异严格为 0；6 项 AUC/PR-AUC 存在 ≤3.7e-06 漂移，经定位为 `roc_auc_score` 数值积分的浮点求和顺序差异，非代码缺陷。所有阈值类指标（accuracy/precision/recall/F1）为 0 差异——即**阈值类指标可逐位复现，排序类指标（AUC/PR-AUC）只可在容差 1e-5 量级内复现**，验收时须按此口径设定容差。

### 3.6 五层递进验收审计机制

**技术要点**：这是本项目最具特色的设计。传统做法是"跑一遍脚本看结果"，本项目构建了一条**不重训、只推理、可被第三方独立执行**的证据链：

| 层      | 内容                                                                  | 排除的失败模式                  |
| ------ | ------------------------------------------------------------------- | ------------------------ |
| **L1** | 用混淆矩阵四格数**手算** Accuracy/Precision/Recall/F1，与 sklearn 结果对拍（容差 1e-6） | 排除库调用错误、API 误用——手算不依赖任何库 |
| **L2** | 校验测试集规模 384、类别分布 185/199、特征数 300、2557 条唯一 ID、ID 前缀与标签 100% 一致       | 数据被篡改、错位、串档              |
| **L3** | 记录 6 个 pkl 的 MD5 指纹                                                 | 审计对象与交付对象不是同一文件          |
| **L4** | 复算指标 vs `model_comparison_data.csv` 记录值逐项比对（30 项，容差 1e-6）           | 交付文档数值与实际模型脱节            |
| **L5** | 按阈值（默认准确率 ≥80%，`--threshold` 可调）给出达标判定与退出码                          | 无法给出明确的合格/不合格结论          |

设计上的三个亮点：

- **L1 的手算对拍**是最有价值的一层——它不依赖 sklearn，若指标算错（比如 precision 与 recall 写反），手算会立刻暴露（`src/99_verify_test_metrics.py:89-101`）；
- **只读不重训**：审计脚本仅做 `pickle.load` + `predict`，不触碰训练代码，因此审计结论不受训练过程 bug 污染；
- **双输出**：`verification_report.json`（机读，供程序化验收）+ `VERIFICATION_SUMMARY.md`（人读，含达标表、指纹表、边界声明）。

实测结果：30/30 项一致，L1 最大差 1.11e-16，L5 全部达标，退出码 0。

### 3.7 SHAP 可解释性分析

**技术要点**：黑箱模型在药物筛选场景难以通过评审。项目采用 SHAP 提供事后可解释性，并做了若干工程化处理：


- **解释器自动适配**：`shap.Explainer(classifier, background)` 自动选择 TreeExplainer / LinearExplainer 等最优解释器，避免手工选型错误（`src/05_shap_analysis.py:168`）；
- **背景集采样**：训练集采样至 500 样本作 background，控制基线计算量；
- **分级降级策略**：主路径失败时自动回退到 `KernelExplainer` + `shap.kmeans` 压缩背景集，并按批（batch=100）处理；SVM 因 KernelExplainer 极慢直接跳过；
- **中间数据全量落盘**：SHAP 值矩阵、全局重要性排名、模型间重要性相关性矩阵、Top20 交集与独有特征，全部保存为 CSV/JSON，使图表可脱离重算独立重绘。

产出包括 4 个模型（LR/RF/XGB/GB）的完整归因数据、模型间重要性相关性矩阵，以及跨模型 Top20 特征交集分析——后者用于验证不同学习机制是否学到一致的生物学信号，是交叉印证模型可信度的有效手段。

**SHAP 结果的引用范围（重要）**

1. **Gradient Boosting 的重要性排名存在计算缺陷。** 其 SHAP 值矩阵原始形态为 `[50, 300, 2]`（3 维），而 `save_intermediate_data` 中的重要性计算路径未做 3D→2D 归约（对比同函数内落盘路径 `src/05_shap_analysis.py:356-364` 做了归约），导致 `np.abs(...).mean(axis=0)` 得到 `(300, 2)` 后被 flatten 成 600 维，与 300 维特征名错位。**后果**：GB 在 `feature_importance_ranking.csv` 中只有 199 行、rank 不连续（最大 295、缺 3–6 等），其排名不可用。GB 实际仅用了 50 个样本（其余模型为 384），样本量本身也偏小。
2. **Top20 交集分析方向写反。** `src/05_shap_analysis.py:432` 用 `np.argsort(mean_abs_shap)[:20]` 取的是**最不重要**的 20 个特征，却将其作为"Top20 特征集"参与交集/独有分析。因此 `model_comparison_summary.json` 中 `common_top20_features` 为空、各模型 `unique_top20_features` 数量异常偏多，该结果**不代表真实的特征共识度**。
3. **模型间重要性相关性矩阵同样受此影响**（`src/05_shap_analysis.py:419` 用同样的 min_len 截断），其数值（如 RF–XGBoost 0.847）不宜直接引用为"跨模型信号一致性"的证据。

> **引用范围**：SHAP 的**单模型**归因数据（`shap_values/*.csv`、LR/XGB/RF 的 `shap_summary.json`）可信；**跨模型对比类结论**（交集、独有性、相关性）在修复上述缺陷前不可作为验收证据。修复方式：`[:20]` 改为 `[-20:]`，并在重要性计算前统一做 `[:, :, 1]` 归约。

### 3.8 SHAP 与统计口径相互印证

基于单模型归因数据，三条独立路径指向同一结论：

```
测试集指标（GB 0.8646，ROC-AUC 0.9432）
        ↑ 解释          ↑ 对照
   SHAP 重要性         5 折 CV（0.8714）
        ↓ 发现            ↓
   300 维入选集 ←── L1 竞赛（0.8899）
```

即 GB/XGB 组成的树模型族泛化能力最强，LR 因高维线性假设而泛化最弱。三条路径数据来源彼此独立，其一致性构成对结论可靠性的交叉支撑。

---

## 四、结论

AMP-ML 项目在 2557 样本、4921 维特征的条件下，通过"三源互补特征体系 + 5 方法竞赛降维至 300 维 + 6 模型嵌套 CV 训练"的建模路径，在独立测试集上取得 **准确率 0.8646 / F1 0.8706 / ROC-AUC 0.9432** 的性能，主推模型 Gradient Boosting 阳性漏检率 12.06%，全部 6 个模型均通过 80% 准确率验收线。

项目真正的工程价值不止于精度数字，而在于**将可复现性与可验收性内建为架构约束**：单一配置事实源消除了配置漂移，依赖全量锁定消除了环境漂移，五层审计脚本提供了第三方可独立执行的证据链（30/30 项数值一致），SHAP 分析为黑箱决策提供了生物学可解释支撑。48 项指标重跑比对中 42 项差异为 0、其余 6 项漂移 ≤3.7e-06 并已定位为浮点求和顺序差异，是这一设计的直接验证。

整体上，本报告所述结论在以下边界内成立：数值口径以独立测试集为准、训练集交叉验证指标仅作辅助参考；可解释性结论限于单模型归因，跨模型对比分析不作为验收依据（3.7 节）；数据构建层的序列获取与去冗余环节不纳入复现承诺范围（1.2、1.7）。上述边界与报告正文各章的结论一一对应，读者可按第 1—3 章的接口契约与验收口径独立复核。

---

*本报告全部数值来源于 `results/` 与 `data/` 下的 CSV/JSON 权威文件，代码论断均给出可定位的文件行号。*
