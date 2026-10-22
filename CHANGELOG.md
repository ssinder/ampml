# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- 负样本采样敏感性分析（`src/07_negative_sampling_sensitivity.py`）：3 种 L0 采样策略 × 6 模型 = 18 组指标，
  首次量化 AMP 分类任务中的负样本分布偏差。

---

## [0.1.0] - 2026-10-22

首个正式发布版本。此版本将原「代码存档快照」重构为**可复现、可验收、可引用**的基准仓库。

### Added

**工程治理**
- `src/config.py` — 路径、随机种子、划分比例、CV 折数、目标特征数的**单一事实来源**
- `src/00_check_environment.py` — 环境自检，复现流水线第一道门禁
- `src/99_verify_test_metrics.py` — 五层独立验收审计，覆盖 30 项数值核对
- `run_all.sh` — 一键复现流水线，支持 `--list` / `--from` / `--only` / `--clean`
- `.gitignore` — 精确区分「可再生大体积中间产物」与「小体积复现凭证」

**可引用性与社区基建**
- `LICENSE`（MIT）、`CITATION.cff`、`CONTRIBUTING.md`、`CODE_OF_CONDUCT.md`、`SECURITY.md`
- `.github/` — CI 流水线、Issue/PR 模板、依赖自动更新、标签体系、stale 机器人
- `pyproject.toml` + `ampml` 命令行入口（`predict` / `benchmark` / `verify` 子命令）
- `notebooks/quickstart.ipynb` — 零训练、秒级出结果的 Colab 快速上手

**文档与结果**
- `docs/TECHNICAL_REPORT.md` + `docs/AMP-ML技术报告.docx` — 技术报告（含 Word 交付版）
- `docs/REPRODUCIBILITY.md` — 复现性报告
- `docs/THIRD_PARTY_VERIFICATION.md` — 第三方验收手册
- `results/VERIFICATION_SUMMARY.md` + `results/verification_report.json` — 审计摘要（人读 / 机读）
- `results/` 10 个指标 CSV + `results/shap/` 完整 SHAP 产物
- `figures/` 4 张出版级图表 + 模型拓扑 SVG

### Changed
- 上游脚本由 `data/` `feature/` 平铺结构归位为 `ampmodel/` 层级布局，
  与脚本内 `PROJECT_ROOT = Path(__file__).parent.parent.parent` 的路径约定保持一致
- 所有业务脚本改为从 `config.py` 导入路径与随机种子，移除硬编码相对路径
- `src/01_data_preparation.py` — 修复样本 ID 回溯逻辑（原实现用 `df.iloc[:len(y_train)]`
  取原始表前 N 行，且赋值后被整表重建覆盖，导致产出 CSV 丢失 ID 列）；
  改为 DataFrame + 索引切分，样本 ID 另存 `data/sample_id_mapping.csv`
- 模型文件位置统一为 `results/models/`（原 `models/` 目录与脚本读取路径不一致，
  导致 `06_voting_ensemble_prediction.py` 与 SHAP 分析必然 FileNotFoundError）
- README 重构为转化漏斗结构：徽章行、结果图、示例输出、方法表、限制声明、引用区块

### Fixed
- 修正 README 中关于特征描述符的错误宣称：PAAC / APAAC / Moran / Geary / NMBroto
  共 5 个描述符在上游计算中实际失败，4921 维为扣除失败项后的实际维度
- 上游脚本中的 Windows 绝对路径（`d:/trae/paper/...`）改为相对项目根目录解析

### Known limitations
- 负样本（L0）为 UniProt 全蛋白 ≤60 aa 片段，与 L1 天然抗菌肽**分布不可比** —— 当前最大方法学局限
- 无外部独立验证集，报告指标均为内部留出集表现
- MMseqs2 聚类参数（identity / coverage / 模式）无记录，上游代表序列无法重生成
- 上游存在 6 条跨类别标签冲突
- 跨模型 SHAP 对比类结论不适用（单模型归因可信）

[Unreleased]: https://github.com/ssinder/ampml/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ssinder/ampml/releases/tag/v0.1.0
