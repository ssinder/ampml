# AMP ML — 结果目录说明

本目录存放模型文件与全部评估中间数据。所有数值均由 `src/` 下脚本生成，可通过重跑复现（见 `docs/REPRODUCIBILITY.md`）。

## 权威数值来源

**以本目录 CSV 文件为准。** 本文档中的表格由 `cv_results_summary.csv` 与 `verification_report.json` 生成，请勿手工编辑数值。

---

## 一、两套指标口径（务必区分）

| 口径 | 用途 | 文件 |
|------|------|------|
| **训练集 5 折交叉验证** | 辅助参考，**不作验收依据** | `cv_results_summary.csv`、`cv_fold_results.csv` |
| **独立测试集** | **验收依据** | `model_comparison_data.csv`、`verification_report.json` |

训练集 CV 指标系统性高于独立测试集，这是正常的（CV 训练样本更多且参与评估的样本与训练同分布）。对外报告准确率时**必须使用测试集口径**。

---

## 二、独立测试集结果（验收口径，n=384）

数据来源：`results/verification_report.json`（2026-10-16 复算，与交付存档 30/30 数值一致）

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|----------|-----------|--------|-----|---------|
| **Gradient Boosting** | **0.8646** | 0.8621 | 0.8794 | **0.8706** | **0.9432** |
| Voting Ensemble | 0.8646 | 0.8657 | 0.8744 | 0.8700 | 0.9369 |
| Random Forest | 0.8464 | 0.8646 | 0.8342 | 0.8491 | 0.9322 |
| SVM | 0.8438 | 0.8564 | 0.8392 | 0.8477 | 0.9132 |
| XGBoost | 0.8438 | 0.8458 | 0.8543 | 0.8500 | 0.9354 |
| Logistic Regression | 0.8203 | 0.8186 | 0.8392 | 0.8288 | 0.9049 |

**主推模型：Gradient Boosting** — 与 Voting Ensemble 准确率并列最高（均 0.8646），但 ROC-AUC（0.9432）与 F1（0.8706）均更优。

**阳性漏检率**（GB）：FN=24 / 阳性总数 199 = **12.06%**
> 注意区分：24/384 = 6.25% 是 FN 占全测试集比例，不是漏检率。

---

## 三、训练集 5 折交叉验证（辅助参考）

数据来源：`cv_results_summary.csv`

| 模型 | Accuracy | F1 | ROC-AUC | PR-AUC |
|------|----------|-----|---------|--------|
| Logistic Regression | 0.9385±0.011 | 0.940 | 0.9867 | 0.988 |
| Voting Ensemble | 0.9078±0.012 | 0.909 | 0.9733 | 0.977 |
| SVM | 0.9245±0.018 | 0.925 | 0.9784 | 0.981 |
| XGBoost | 0.8720±0.012 | 0.874 | 0.9379 | 0.950 |
| Gradient Boosting | 0.8714±0.013 | 0.874 | 0.9375 | 0.950 |
| Random Forest | 0.8603±0.009 | 0.861 | 0.9341 | 0.947 |

> 注：LR 在 CV 上表现最好，但在独立测试集上最差（0.8203）—— 说明其泛化能力最弱，交叉验证分数不可作为模型选择依据。

---

## 四、文件清单

### 模型文件 `/models/`
pickle 格式，由 `03_model_training_cv.py` 生成（Python 3.13.9 + `requirements.txt` 锁定版本）

- `logistic_regression_cv_model.pkl`
- `random_forest_cv_model.pkl`
- `svm_cv_model.pkl`
- `xgboost_cv_model.pkl`
- `gradient_boosting_cv_model.pkl`  ← 主推模型
- `voting_ensemble_cv_model.pkl`  ← 备选

### 交叉验证结果
- `cv_results_summary.csv` — 各模型 CV 指标汇总（均值±标准差）
- `cv_fold_results.csv` — 逐折明细（6 模型 × 5 折 × 9 指标）
- `cv_roc_curve_data.csv` — CV ROC 曲线点
- `cv_precision_recall_curve_data.csv` — CV PR 曲线点
- `cv_confusion_matrix_data.csv` — CV 混淆矩阵

### 测试集结果
- `model_comparison_data.csv` — 六模型测试集对比
- `roc_curve_data.csv` / `precision_recall_curve_data.csv` / `confusion_matrix_data.csv` — 作图用中间数据
- `predictions.csv` — 逐样本预测明细（由 `06_voting_ensemble_prediction.py` 生成）
- `verification_report.json` — 验收审计报告（30/30 数值复算一致）

### SHAP 可解释性 `/shap/`
- `feature_importance_ranking.csv` — 全局特征重要性排名
- `feature_names.csv` — 300 个入选特征列表
- `model_comparison_summary.json` — SHAP 视角的模型对比
- `model_correlation_matrix.csv` — 模型预测相关性
- `shap_summary.json` — 各模型 Top 特征
- `shap_results_info.json` — SHAP 计算元数据
- `test_features.csv` — 测试集特征副本
- `shap_values/` — 各模型 SHAP 值明细

### 元数据
- `training_metadata.json` — 训练配置（随机种子 42、5 折、6 模型）

---

## 五、Top 重要特征（SHAP 分析）

数据源：`shap/shap_summary.json`（2026-10-20 核验）。

| 模型 | Top 1 | Top 2 | Top 3 |
|------|-------|-------|-------|
| Logistic Regression | `M` (0.4569) | `Isoelectric_Point` (0.3288) | `KyteDoolittle_Moment` (0.2972) |
| XGBoost | `KyteDoolittle_Moment` (0.5304) | `CTDC_hydrophobicity_ARGP820101.G1` (0.4973) | `CTDD_charge.3.residue100` (0.3706) |
| Random Forest | `CTDC_hydrophobicity_ARGP820101.G1` (0.0283) | `Isoelectric_Point` (0.0281) | `CTDT_charge.Tr1221` (0.0271) |

> **引用限制**：单模型归因（`shap_values/`、上表的 `shap_summary.json`）可信；**跨模型对比类结论不可引用**——`05_shap_analysis.py:432` 的 Top20 集合用 `argsort(...)[:20]` 取的是最不重要的 20 个特征，且重要性计算未对 Gradient Boosting 的 3 维 SHAP 值做归约，导致其排名错位（仅 199 行、rank 不连续）。详见 `docs/TECHNICAL_REPORT.md` 第 3.7 节。

---

## 六、复现方式

```bash
python3.13 -m venv venv
./venv/bin/pip install -r requirements.txt

python src/01_data_preparation.py      # → data/
python src/02_feature_selection.py     # → data/*_selected.csv
python src/03_model_training_cv.py     # → results/
python src/03b_generate_cv_curves.py
python src/04b_generate_figures_optimized.py
python src/05_shap_analysis.py
python src/99_verify_test_metrics.py   # 复算验收指标
```

**已实测**：重跑 `03_model_training_cv.py` 与本目录存档比对 48 项指标 ——
accuracy / precision / recall / F1 / balanced_accuracy / MCC 共 42 项差异为 **0**；
ROC-AUC / PR-AUC 共 6 项漂移 ≤ 3.7e-06（数值积分浮点误差，非代码缺陷）。
详见 `docs/REPRODUCIBILITY.md`。
