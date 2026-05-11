# AMPML 实验方法与脚本逻辑文档

## 1. 概述

本项目(AMPML)是一个抗菌肽(Antimicrobial Peptide, AMP)机器学习分类流程,包含完整的数据处理、特征选择、模型训练、交叉验证评估和可解释性分析流程。

### 实验流程总览

```
数据加载 → 数据划分 → 特征选择 → 模型训练 → 交叉验证 → 可解释性分析 → 预测
```

---

## 2. 脚本功能详解

### 2.1 数据准备 (01_data_preparation.py)

**功能**: 加载预处理后的特征数据,提取特征和标签,划分训练/验证/测试集

**主要功能**:

1. **数据加载**: 从 `data/features_processed.csv` 加载预处理后的数据
2. **标签提取**: 从ID列(如 L0_xxxxxx, L1_xxxxxx)中提取分类标签,L0=非抗菌肽(0),L1=抗菌肽(1)
3. **数据集划分**: 采用分层采样保持类别比例
   - 训练集: 70%
   - 验证集: 15%
   - 测试集: 15%
4. **缺失值处理**: 使用中位数填充策略(对异常值鲁棒)
5. **数据质量验证**: 检查特征数量、样本数量、标签分布、无穷值、常量特征等

**关键参数**:
- `RANDOM_STATE`: 42 (随机种子)
- `test_size`: 0.3 (第一次划分比例)
- `test_size`: 0.5 (第二次划分比例,将临时集均分)
- `SimpleImputer(strategy='median')`: 中位数填充

**输出文件**:
- `data/train_data.csv` - 训练数据
- `data/val_data.csv` - 验证数据
- `data/test_data.csv` - 测试数据
- `data/feature_info.npy` - 特征信息

---

### 2.2 特征选择 (02_feature_selection.py)

**功能**: 对高维特征进行选择,降低维度同时保留重要特征

**主要方法**:

| 方法 | 描述 | 关键参数 |
|------|------|----------|
| **方差阈值** | 移除方差低于阈值的特征 | `threshold=0.01` |
| **L1正则化逻辑回归** | 使用L1正则化使部分系数为0 | `C=0.5, solver='saga', max_features=200` |
| **随机森林特征重要性** | 基于随机森林的特征重要性 | `n_estimators=100, max_depth=10, max_features=200` |
| **SelectKBest** | 基于ANOVA F值选择top-k特征 | `k=150, score_func=f_classif` |

**评估方法**:
- 使用5折交叉验证评估各方法的ROC-AUC分数
- 交叉验证参数: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 自动选择表现最好的方法(默认使用L1逻辑回归结果)

**具体参数配置**:
```python
# L1正则化逻辑回归
selector = SelectFromModel(
    LogisticRegression(penalty='l1', solver='saga', C=0.5, max_iter=2000),
    threshold='median',
    max_features=200
)

# 随机森林
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    threshold='median',
    max_features=200
)

# SelectKBest
selector = SelectKBest(f_classif, k=150)
```

**输出文件**:
- `data/selected_features.csv` - 选中的特征列表(约200个)
- 特征选择元数据(JSON格式)

---

### 2.3 模型训练与交叉验证 (03_model_training_cv.py)

**功能**: 训练多个机器学习模型并进行完整的5折交叉验证评估

**包含模型**:

| 模型 | Pipeline配置 | 关键超参数搜索空间 |
|------|-------------|-------------------|
| **Logistic Regression** | StandardScaler + LogisticRegression | `C: uniform(0.001,100)`, `penalty: l2`, `solver: lbfgs`, `class_weight: [None, 'balanced']` |
| **Random Forest** | 直接使用(无需标准化) | `n_estimators: [100,200,300]`, `max_depth: [5,10,15,20,None]`, `min_samples_split: [2,5,10]`, `min_samples_leaf: [1,2,4]`, `max_features: ['sqrt','log2']` |
| **SVM** | StandardScaler + SVC | `C: uniform(0.1,100)`, `kernel: ['rbf','linear']`, `gamma: ['scale','auto']`, `class_weight: [None,'balanced']` |
| **XGBoost** | 直接使用 | `n_estimators: [100,200,300]`, `max_depth: [3,5,7]`, `learning_rate: [0.01,0.05,0.1,0.2]`, `subsample: [0.6,0.8,1.0]`, `colsample_bytree: [0.6,0.8,1.0]`, `min_child_weight: [1,3,5]`, `gamma: [0,0.1,0.2]` |
| **Gradient Boosting** | 直接使用 | `n_estimators: [100,200]`, `max_depth: [3,5,7]`, `learning_rate: [0.05,0.1,0.2]`, `subsample: [0.8,1.0]`, `min_samples_split: [2,5]`, `min_samples_leaf: [1,2]` |
| **Voting Ensemble** | StandardScaler + VotingClassifier | `voting: 'soft'`, 集成LogisticRegression, RandomForest, XGBoost |

**交叉验证策略**:
- 外层: 5折分层交叉验证 (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`)
- 超参数搜索: `RandomizedSearchCV(n_iter=30, scoring='roc_auc')`
- 嵌套交叉验证: 3折内层CV用于超参数搜索,5折外层CV用于评估

**评估指标**:

| 指标 | 描述 |
|------|------|
| Accuracy | 准确率 |
| Precision | 精确率 |
| Recall | 召回率 |
| F1-Score | F1分数 |
| ROC-AUC | ROC曲线下面积 |
| PR-AUC | 精确率-召回率曲线下面积 (Average Precision) |
| Balanced Accuracy | 平衡准确率 |
| MCC | Matthews相关系数 |

**优化策略**:

- 使用 `Pipeline` 封装预处理步骤,防止数据泄漏
- 使用 `sklearn.base.clone()` 进行模型克隆
- 超参数搜索使用 `RandomizedSearchCV`
- 嵌套交叉验证评估(Nested CV)验证泛化性能
- XGBoost参数: `eval_metric='logloss', verbosity=0`

**输出文件**:
- `results/models/*.pkl` - 各模型文件
- `results/cv_results_summary.csv` - 交叉验证汇总结果
- `results/cv_fold_results.csv` - 各折详细结果
- `results/cv_roc_curve_data.csv` - CV ROC曲线数据
- `results/cv_precision_recall_curve_data.csv` - CV PR曲线数据
- `results/training_metadata.json` - 训练元数据

---

### 2.4 CV曲线数据生成 (03b_generate_cv_curves.py)

**功能**: 从已训练的交叉验证模型生成ROC曲线、PR曲线和混淆矩阵数据

**主要流程**:

1. 加载训练数据和已保存的模型
2. 对5折交叉验证的每个fold进行预测,收集所有验证集的预测结果
3. 计算整体ROC曲线、PR曲线和混淆矩阵
4. 保存为CSV文件供后续可视化使用

**关键参数**:
- `RANDOM_STATE`: 42
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 阈值: 0.5 (用于二分类预测)

**输出文件**:
- `results/cv_roc_curve_data.csv` - 包含字段: Model, FPR, TPR, Threshold, AUC
- `results/cv_precision_recall_curve_data.csv` - 包含字段: Model, Precision, Recall, Threshold, AP
- `results/cv_confusion_matrix_data.csv` - 包含字段: Model, TN, FP, FN, TP, Threshold

---

### 2.5 图表生成 (04b_generate_figures_optimized.py)

**功能**: 生成顶刊级别的高质量可视化图表

**配色方案**:
```python
COLORS = {
    'Logistic Regression': '#2E86AB',
    'Random Forest': '#E94F37',
    'SVM': '#1B998B',
    'XGBoost': '#F39237',
    'Gradient Boosting': '#8E44AD',
    'Voting Ensemble': '#27AE60'
}
```

**Matplotlib参数** (顶刊要求):
```python
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3
})
```

**生成的图表**:

1. **ROC曲线 (Figure 1)**:
   - 多模型ROC曲线对比
   - 包含AUC值标注
   - 基准线: 黑色虚线 (Random Classifier)
   - X轴: False Positive Rate
   - Y轴: True Positive Rate

2. **混淆矩阵 (Figure 2)**:
   - 各模型的混淆矩阵可视化
   - 显示标签: ['Non-AMP', 'AMP']
   - 颜色映射: Blues

3. **精确率-召回率曲线 (Figure 3)**:
   - PR曲线对比
   - 包含AP值标注

4. **模型性能对比 (Figure 4)**:
   - 多指标柱状图对比

**输出文件** (300 DPI, PNG格式):
- `figures/roc_curves.png`
- `figures/confusion_matrices.png`
- `figures/precision_recall_curves.png`
- `figures/model_comparison.png`

---

### 2.6 SHAP可解释性分析 (05_shap_analysis.py)

**功能**: 使用SHAP(SHapley Additive exPlanations)进行模型可解释性分析

**分析对象**:
- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
- SVM

**性能优化配置**:
```python
MAX_SAMPLES_FOR_SHAP = 500   # 大数据集采样限制
MAX_SAMPLES_FOR_SVM = 50    # SVM特别限制(KernelExplainer很慢)
BATCH_SIZE = 100            # 批量处理大小
```

**SHAP计算策略**:
- 自动检测最佳Explainer类型 (`shap.Explainer`)
- 大数据集自动采样
- SVM使用KernelExplainer回退方案

**关键参数**:
- Background数据采样: 使用训练集,最多500样本
- 测试数据采样: 最多500样本(SVM限制为50)
- 输出格式: 二分类取正类(class 1)SHAP值

**输出文件** (`results/shap/`):
- 各模型的SHAP值矩阵
- 特征重要性排序
- 依赖图数据

---

### 2.7 集成模型预测 (06_voting_ensemble_prediction.py)

**功能**: 使用训练好的Voting Ensemble模型对新数据进行预测

**主要流程**:

1. 加载训练好的集成模型 (`voting_ensemble_cv_model.pkl`)
2. 加载选中的特征列表 (`selected_features.csv`)
3. 加载并筛选特征数据
4. 进行预测(分类标签+概率)
5. 保存预测结果

**预测输出字段**:
- 原始特征 (前10个)
- true_label (如有)
- predicted_label
- probability_class_0 (非抗菌肽概率)
- probability_class_1 (抗菌肽概率)

**输出文件**:
- `results/predictions.csv`

---

## 3. 执行顺序与依赖关系

```
┌─────────────────────────────────────────────────────────────────────┐
│                         数据准备阶段                                 │
├─────────────────────────────────────────────────────────────────────┤
│  01_data_preparation.py                                            │
│       ↓                                                            │
│  输入: data/features_processed.csv                                  │
│  输出: train_data.csv, val_data.csv, test_data.csv                 │
│  参数: test_size=0.3, stratify=y, random_state=42                  │
│        imputer=SimpleImputer(strategy='median')                    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         特征选择阶段                                 │
├─────────────────────────────────────────────────────────────────────┤
│  02_feature_selection.py                                           │
│       ↓                                                            │
│  输入: train_data.csv, val_data.csv, test_data.csv                  │
│  输出: selected_features.csv                                        │
│  方法: L1 Logistic Regression (C=0.5, max_features=200)           │
│        交叉验证评估: 5-fold StratifiedKFold                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         模型训练阶段                                 │
├─────────────────────────────────────────────────────────────────────┤
│  03_model_training_cv.py                                           │
│       ↓                                                            │
│  输入: train_data_selected.csv, val_data_selected.csv,             │
│        test_data_selected.csv                                       │
│  输出: 各模型pkl文件, CV结果CSV                                    │
│  模型: LR, RF, SVM, XGBoost, GB, Voting Ensemble                  │
│  CV: 5-fold, nested CV, RandomizedSearchCV(n_iter=30)              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    曲线数据生成 & 可视化                             │
├─────────────────────────────────────────────────────────────────────┤
│  03b_generate_cv_curves.py  →  04b_generate_figures_optimized.py   │
│  (生成曲线数据)                    (生成图表)                      │
│  参数: threshold=0.5              DPI=300, 顶刊配色                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         可解释性分析                                │
├─────────────────────────────────────────────────────────────────────┤
│  05_shap_analysis.py                                               │
│       ↓                                                            │
│  输入: 模型文件, 测试数据                                           │
│  输出: SHAP分析结果                                                │
│  参数: max_samples=500, SVM限制50                                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         模型预测                                    │
├─────────────────────────────────────────────────────────────────────┤
│  06_voting_ensemble_prediction.py                                  │
│       ↓                                                            │
│  输入: 集成模型, 新数据                                             │
│  输出: predictions.csv                                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 关键技术点详细说明

### 4.1 数据预处理

**分层采样 (Stratified Split)**:
```python
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.3, 
    stratify=y,  # 保持类别比例
    random_state=42
)
# 第二次划分
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5, 
    stratify=y_temp,
    random_state=42
)
```

**缺失值处理**:
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
```

### 4.2 特征工程

**Pipeline封装**:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
```

**多方法对比评估**:
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# L1逻辑回归选择
selector = SelectFromModel(
    LogisticRegression(penalty='l1', solver='saga', C=0.5),
    max_features=200
)
# 5折CV评估
scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
```

### 4.3 模型训练

**超参数搜索**:
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    'classifier__C': uniform(0.001, 100),
    'classifier__n_estimators': randint(100, 300),
    'classifier__max_depth': [3, 5, 7, None]
}

random_search = RandomizedSearchCV(
    pipeline, 
    param_distributions, 
    n_iter=30,  # 迭代30次
    cv=5,        # 5折CV
    scoring='roc_auc',
    n_jobs=-1    # 并行计算
)
```

**嵌套交叉验证**:
```python
# 外层: 评估
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
# 内层: 超参数搜索
cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
```

### 4.4 可解释性

**SHAP分析**:
```python
import shap
explainer = shap.Explainer(model, X_background)
shap_values = explainer(X_test)
# 二分类: 取正类SHAP值
shap_values_positive = shap_values.values[:, :, 1]
```

---

## 5. 运行说明

### 前置条件
```bash
# 安装必要依赖
pip install pandas numpy scikit-learn xgboost shap matplotlib scipy
```

### 执行流程
```bash
# 1. 数据准备
python src/01_data_preparation.py

# 2. 特征选择
python src/02_feature_selection.py

# 3. 模型训练与交叉验证
python src/03_model_training_cv.py

# 4. 生成CV曲线数据
python src/03b_generate_cv_curves.py

# 5. 生成可视化图表
python src/04b_generate_figures_optimized.py

# 6. SHAP可解释性分析
python src/05_shap_analysis.py

# 7. 集成模型预测(可选)
python src/06_voting_ensemble_prediction.py
```

---

## 6. 输出目录结构

```
ampml/
├── data/                          # 数据目录
│   ├── features_processed.csv    # 预处理后的特征
│   ├── train_data.csv             # 训练数据
│   ├── val_data.csv               # 验证数据
│   ├── test_data.csv              # 测试数据
│   ├── train_data_selected.csv    # 特征选择后的训练数据
│   ├── val_data_selected.csv      # 特征选择后的验证数据
│   ├── test_data_selected.csv     # 特征选择后的测试数据
│   ├── selected_features.csv       # 选中的特征列表(~200个)
│   └── feature_info.npy           # 特征信息
├── results/                       # 结果目录
│   ├── models/                    # 训练好的模型
│   │   ├── logistic_regression_cv_model.pkl
│   │   ├── random_forest_cv_model.pkl
│   │   ├── svm_cv_model.pkl
│   │   ├── xgboost_cv_model.pkl
│   │   ├── gradient_boosting_cv_model.pkl
│   │   └── voting_ensemble_cv_model.pkl
│   ├── cv_results_summary.csv     # CV结果汇总(8个指标)
│   ├── cv_fold_results.csv         # 各折详细结果
│   ├── cv_roc_curve_data.csv      # CV ROC曲线数据
│   ├── cv_precision_recall_curve_data.csv
│   ├── cv_confusion_matrix_data.csv
│   ├── training_metadata.json     # 训练元数据
│   ├── predictions.csv            # 预测结果
│   └── shap/                      # SHAP分析结果
├── figures/                       # 可视化图表
│   ├── roc_curves.png             # ROC曲线
│   ├── confusion_matrices.png     # 混淆矩阵
│   ├── precision_recall_curves.png # PR曲线
│   └── model_comparison.png      # 模型对比
└── src/                           # 源代码
    ├── 01_data_preparation.py     # 数据准备
    ├── 02_feature_selection.py   # 特征选择
    ├── 03_model_training_cv.py    # 模型训练+CV
    ├── 03b_generate_cv_curves.py  # CV曲线数据
    ├── 04b_generate_figures_optimized.py # 图表生成
    ├── 05_shap_analysis.py        # SHAP分析
    └── 06_voting_ensemble_prediction.py # 预测
```

---

## 7. 完整参数配置汇总

### 全局配置
| 参数 | 值 | 说明 |
|------|-----|------|
| RANDOM_STATE | 42 | 随机种子 |
| CV_FOLDS | 5 | 交叉验证折数 |
| MAX_FEATURES | 200 | 特征选择最大特征数 |

### 特征选择参数
| 方法 | 参数 |
|------|------|
| 方差阈值 | threshold=0.01 |
| L1逻辑回归 | C=0.5, solver='saga', max_iter=2000 |
| 随机森林 | n_estimators=100, max_depth=10 |
| SelectKBest | k=150, score_func=f_classif |

### 模型超参数搜索空间
| 模型 | 搜索参数 |
|------|----------|
| LR | C: uniform(0.001,100), class_weight: [None, 'balanced'] |
| RF | n_estimators: [100,200,300], max_depth: [5,10,15,20,None] |
| SVM | C: uniform(0.1,100), kernel: ['rbf','linear'], gamma: ['scale','auto'] |
| XGBoost | n_estimators: [100,200,300], max_depth: [3,5,7], learning_rate: [0.01,0.05,0.1,0.2] |
| GB | n_estimators: [100,200], max_depth: [3,5,7], learning_rate: [0.05,0.1,0.2] |

### SHAP参数
| 参数 | 值 |
|------|-----|
| MAX_SAMPLES_FOR_SHAP | 500 |
| MAX_SAMPLES_FOR_SVM | 50 |
| BATCH_SIZE | 100 |

---

## 8. 总结

本项目提供了一个完整的抗菌肽机器学习分类流程,从原始数据到最终预测,涵盖了:

1. **数据处理**: 分层划分、缺失值中位数填充、数据质量验证
2. **特征工程**: 4种特征选择方法,5折CV评估,选择约200个最优特征
3. **模型训练**: 6种机器学习模型,完整超参数优化, RandomizedSearchCV(n_iter=30)
4. **模型评估**: 8项评估指标,5折交叉验证,嵌套交叉验证
5. **可视化**: 顶刊级别图表,专业学术配色,300 DPI高清输出
6. **可解释性**: SHAP分析,特征贡献可视化,自动采样优化性能
7. **预测部署**: 集成模型对新数据进行预测,输出概率和标签

整个流程模块化设计,各脚本职责清晰,可独立运行或组合使用。