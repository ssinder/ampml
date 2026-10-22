# AMP分类模型测试脚本集

本目录包含用于抗菌肽(AMP)分类模型的特征选择、模型训练和结果可视化脚本。

## 目录结构

### 1. 数据准备脚本

| 脚本 | 功能描述 |
|------|----------|
| `prepare_data.py` | 样本提取与划分脚本。从原始数据中提取少量样本，划分独立测试集，使用重复分层交叉验证(5折×3次重复)划分训练/验证集 |

### 2. 特征选择脚本

| 脚本 | 功能描述 |
|------|----------|
| `feature_selection.py` | 综合特征选择流水线。集成方差阈值过滤、SelectKBest、随机森林重要性、组合特征选择等多种方法 |
| `elasticnet_feature_selection.py` | ElasticNet正则化特征选择。使用LogisticRegression + ElasticNet惩罚，通过GridSearchCV优化超参数，选择非零系数特征 |
| `randomforest_feature_selection.py` | 随机森林特征重要性选择。基于随机森林的feature_importances_进行特征排序和筛选 |
| `rfe_feature_selection.py` | 递归消除特征选择(RFE)。使用随机森林作为基模型，递归消除不重要特征，获得排名 |
| `statistical_feature_selection.py` | 统计检验特征选择。使用ANOVA F检验(SelectKBest + f_classif)评估特征与标签的相关性 |

### 3. 模型训练脚本

| 脚本 | 功能描述 |
|------|----------|
| `model_training.py` | 多模型训练与评估。支持10种分类模型(LogisticRegression, RandomForest, GradientBoosting, SVM-RBF, SVM-Linear, KNN, NaiveBayes, DecisionTree, AdaBoost, MLP)，使用重复交叉验证评估 |
| `multi_feature_training.py` | 多特征数训练比较。分别使用50/100/200个特征进行训练，比较不同特征数对模型性能的影响 |

### 4. 快速测试脚本

| 脚本 | 功能描述 |
|------|----------|
| `test_randomforest.py` | 随机森林特征选择快速测试 |
| `test_elasticnet.py` | ElasticNet特征选择快速测试 |
| `test_rfe.py` | RFE特征选择快速测试 |
| `test_svm.py` | SVM模型性能测试 |
| `test_statistical.py` | 统计检验特征选择快速测试 |

### 5. 可视化脚本

| 脚本 | 功能描述 |
|------|----------|
| `plot_results.py` | 结果可视化。生成特征数比较图、性能热力图、模型排名图等 |
| `shap_analysis.py` | SHAP可解释性分析。生成beeswarm图、bar图、waterfall图、scatter图、热力图等，分析特征对预测的贡献 |
| `publication_plots.py` | 顶刊级可视化。遵循Nature/Science期刊规范，生成PDF/PNG双格式高质量图表，含6张标准图表 |

## 典型工作流程

```
1. 数据准备
   prepare_data.py
   ↓
2. 特征选择（任选一种或多种）
   feature_selection.py / elasticnet_feature_selection.py /
   randomforest_feature_selection.py / rfe_feature_selection.py /
   statistical_feature_selection.py
   ↓
3. 模型训练
   model_training.py / multi_feature_training.py
   ↓
4. 结果可视化
   plot_results.py / shap_analysis.py / publication_plots.py
```

## 依赖库

- Python 3.x
- numpy, pandas
- scikit-learn
- matplotlib, seaborn
- shap
- scipy


