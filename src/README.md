# AMPML 抗菌肽机器学习预测系统

本目录包含完整的抗菌肽(AMP)机器学习分类流程脚本。

## 脚本执行顺序

```
01_data_preparation.py          → 02_feature_selection.py
         ↓                               ↓
         └───── 03_model_training_cv.py ────┘
                        ↓
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
   03b_generate    04b_generate      05_shap_analysis
   _cv_curves      _figures          _optimized
                        ↓
         ┌───────────────┴───────────────┐
         ↓                               ↓
    06_voting_ensemble_prediction.py   (生成顶刊图表)
```

## 脚本说明

### 1. 数据预处理 (`01_data_preparation.py`)
- **功能**: 加载原始特征数据，进行规范预处理
- **主要步骤**:
  - 从ID列提取标签(L0负例/L1正例)
  - 缺失值检测与中位数填充
  - IQR方法异常值检测
  - 特征标准化(StandardScaler)
  - 数据划分: 70%训练 / 15%验证 / 15%测试
- **输出**: `data/train_data.csv`, `val_data.csv`, `test_data.csv`

### 2. 特征选择 (`02_feature_selection.py`)
- **功能**: 从高维特征中筛选重要特征，降低维度
- **方法**:
  - L1正则化逻辑回归
  - 随机森林特征重要性
  - 互信息(Mutual Information)
  - ANOVA F-test (SelectKBest)
  - SelectPercentile
  - RFECV (递归特征消除+交叉验证)
- **自动选择**: 基于CV ROC-AUC自动选择最佳方法
- **输出**: `data/selected_features.csv`

### 3. 模型训练与交叉验证 (`03_model_training_cv.py`)
- **功能**: 训练多个分类模型并进行完整评估
- **模型**:
  - Logistic Regression
  - Random Forest
  - SVM
  - XGBoost
  - Gradient Boosting
  - Voting Ensemble (Soft Voting)
- **评估**:
  - 5折分层交叉验证
  - 超参数随机搜索(RandomizedSearchCV)
  - 嵌套交叉验证
  - 多指标: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC
- **输出**: `results/models/*.pkl`, `cv_results_summary.csv`

### 4. CV曲线数据生成 (`03b_generate_cv_curves.py`)
- **功能**: 为已训练模型生成ROC/PR曲线数据
- **输出**: `results/cv_roc_curve_data.csv`, `cv_precision_recall_curve_data.csv`

### 5. 可视化图表生成 (`04b_generate_figures_optimized.py`)
- **功能**: 生成顶刊级别可视化图表
- **图表**:
  - Fig1: ROC曲线
  - Fig2: 混淆矩阵
  - Fig3: Precision-Recall曲线
  - Fig4: 模型性能对比柱状图
- **输出**: `figures/fig1_roc_curves.png` 等

### 6. SHAP可解释性分析 (`05_shap_analysis.py`)
- **功能**: 计算模型SHAP值，分析特征重要性
- **输出**:
  - `results/shap/shap_summary.json` - 汇总结果
  - `results/shap/feature_importance_ranking.csv` - 特征排名
  - `results/shap/model_comparison_summary.json` - 模型对比

### 7. 集成模型预测 (`06_voting_ensemble_prediction.py`)
- **功能**: 使用训练好的Voting Ensemble模型对新数据预测
- **输出**: `results/predictions.csv`

## 数据目录结构

```
ampml-main/
├── rawdata/
│   └── all_features_combined_standard.csv   # 原始数据
├── data/
│   ├── train_data.csv / val_data.csv / test_data.csv   # 预处理后
│   ├── selected_features.csv                              # 选中特征
│   ├── train_data_selected.csv / ...                      # 特征选择后
│   └── preprocessing_metadata.json                        # 预处理元数据
├── results/
│   ├── models/                    # 训练好的模型
│   ├── cv_results_summary.csv   # CV评估结果
│   ├── shap/                     # SHAP分析结果
│   └── *.csv                     # 各种曲线数据
└── figures/                      # 生成的可视化图表
```

## 依赖库

```
pandas, numpy, scikit-learn, scipy, xgboost, matplotlib, shap
```

## 使用示例

```bash
# 按顺序执行各脚本
python 01_data_preparation.py
python 02_feature_selection.py
python 03_model_training_cv.py
python 03b_generate_cv_curves.py
python 04b_generate_figures_optimized.py
python 05_shap_analysis.py
python 06_voting_ensemble_prediction.py
```
