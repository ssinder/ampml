"""
模型训练脚本 - 带完整5折交叉验证评估 (优化版)
功能：
1. 使用5折交叉验证进行超参数搜索
2. 对每个模型进行完整的5折交叉验证评估
3. 输出交叉验证的详细统计结果
优化内容：
1. 使用 Pipeline 封装预处理步骤，防止数据泄漏
2. 使用 sklearn.base.clone() 进行模型克隆
3. 修复 XGBoost 已废弃参数
4. 添加 PR-AUC (Average Precision) 指标
5. 添加嵌套交叉验证评估
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import time
import json
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import clone
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix, precision_recall_curve, average_precision_score,
    roc_curve
)
from scipy.stats import uniform, randint
import xgboost as xgb
import os

TRAIN_PATH = "data/train_data_selected.csv"
VAL_PATH = "data/val_data_selected.csv"
TEST_PATH = "data/test_data_selected.csv"
OUTPUT_DIR = "results"
MODEL_DIR = "results/models"
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """加载特征选择后的数据"""
    print("=" * 70)
    print("加载数据")
    print("=" * 70)
    
    train_data = pd.read_csv(TRAIN_PATH)
    val_data = pd.read_csv(VAL_PATH)
    test_data = pd.read_csv(TEST_PATH)
    
    y_train = train_data.iloc[:, -1]
    y_val = val_data.iloc[:, -1]
    y_test = test_data.iloc[:, -1]
    
    X_train = train_data.iloc[:, :-1]
    X_val = val_data.iloc[:, :-1]
    X_test = test_data.iloc[:, :-1]
    
    print(f"训练集: {X_train.shape}")
    print(f"验证集: {X_val.shape}")
    print(f"测试集: {X_test.shape}")
    print(f"类别分布 - 训练集: {dict(y_train.value_counts())}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_pipeline(model, use_scaled=True):
    """创建包含标准化的 Pipeline"""
    if use_scaled:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline([
            ('classifier', model)
        ])
    return pipeline


def get_cv_strategy():
    """获取5折分层交叉验证策略"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cv


def cv_evaluate_model(pipeline, X, y, model_name, cv, use_proba=True):
    """
    对模型进行完整的5折交叉验证评估
    返回每个fold的评估指标
    使用 clone() 进行模型克隆
    """
    print(f"\n  正在进行5折交叉验证评估...")
    
    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc': [],
        'balanced_accuracy': [],
        'mcc': [],
        'pr_auc': []
    }
    
    fold_idx = 1
    for train_idx, val_idx in cv.split(X, y):
        if isinstance(X, pd.DataFrame):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        
        if isinstance(y, pd.Series):
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        # 使用 clone() 克隆模型
        pipeline_clone = clone(pipeline)
        pipeline_clone.fit(X_fold_train, y_fold_train)
        
        y_pred = pipeline_clone.predict(X_fold_val)
        
        fold_results['accuracy'].append(accuracy_score(y_fold_val, y_pred))
        fold_results['precision'].append(precision_score(y_fold_val, y_pred, zero_division=0))
        fold_results['recall'].append(recall_score(y_fold_val, y_pred, zero_division=0))
        fold_results['f1_score'].append(f1_score(y_fold_val, y_pred, zero_division=0))
        fold_results['balanced_accuracy'].append(balanced_accuracy_score(y_fold_val, y_pred))
        fold_results['mcc'].append(matthews_corrcoef(y_fold_val, y_pred))
        
        if use_proba:
            y_proba = pipeline_clone.predict_proba(X_fold_val)[:, 1]
            fold_results['roc_auc'].append(roc_auc_score(y_fold_val, y_proba))
            fold_results['pr_auc'].append(average_precision_score(y_fold_val, y_proba))
        
        print(f"    Fold {fold_idx} 完成")
        fold_idx += 1
    
    return fold_results


def nested_cv_evaluate(pipeline, X, y, cv_outer, model_name):
    """嵌套交叉验证评估 (外层用于评估，内层用于超参数搜索)"""
    print(f"\n  进行嵌套交叉验证评估...")
    
    outer_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_outer.split(X, y)):
        if isinstance(X, pd.DataFrame):
            X_outer_train, X_outer_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_outer_train, X_outer_val = X[train_idx], X[val_idx]
        
        if isinstance(y, pd.Series):
            y_outer_train, y_outer_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            y_outer_train, y_outer_val = y[train_idx], y[val_idx]
        
        # 内层交叉验证进行超参数搜索 (简化版，使用默认参数)
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        # 使用简化的超参数搜索
        try:
            from sklearn.model_selection import GridSearchCV
            
            # 获取分类器
            classifier = pipeline.named_steps['classifier']
            param_grid = {}
            
            # 根据模型类型设置简化的参数网格
            if isinstance(classifier, LogisticRegression):
                param_grid = {'classifier__C': [0.1, 1.0, 10.0]}
            elif isinstance(classifier, RandomForestClassifier):
                param_grid = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [10, 20]}
            elif isinstance(classifier, SVC):
                param_grid = {'classifier__C': [0.1, 1.0, 10.0]}
            elif isinstance(classifier, xgb.XGBClassifier):
                param_grid = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [3, 5]}
            elif isinstance(classifier, GradientBoostingClassifier):
                param_grid = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [3, 5]}
            
            if param_grid:
                grid_search = GridSearchCV(pipeline, param_grid, cv=cv_inner, scoring='roc_auc', n_jobs=-1)
                grid_search.fit(X_outer_train, y_outer_train)
                best_pipeline = grid_search.best_estimator_
            else:
                best_pipeline = clone(pipeline)
                best_pipeline.fit(X_outer_train, y_outer_train)
        except Exception as e:
            print(f"    嵌套CV警告: {e}")
            best_pipeline = clone(pipeline)
            best_pipeline.fit(X_outer_train, y_outer_train)
        
        y_proba = best_pipeline.predict_proba(X_outer_val)[:, 1]
        auc = roc_auc_score(y_outer_val, y_proba)
        outer_scores.append(auc)
        
        print(f"    外层 Fold {fold_idx + 1} AUC: {auc:.4f}")
    
    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)
    print(f"  嵌套CV结果: {mean_score:.4f} ± {std_score:.4f}")
    
    return mean_score, std_score, outer_scores


def train_and_evaluate_with_cv(X_train, y_train, model_name, param_distributions, 
                                 n_iter=30, use_scaled=True):
    """
    训练模型并进行完整的5折交叉验证评估
    使用 Pipeline 封装预处理
    """
    print(f"\n{'='*70}")
    print(f"模型: {model_name}")
    print("=" * 70)
    
    start_time = time.time()
    
    cv = get_cv_strategy()
    
    # 创建基础模型
    if model_name == "Logistic Regression":
        base_model = LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1, max_iter=3000)
    elif model_name == "Random Forest":
        base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == "SVM":
        base_model = SVC(random_state=RANDOM_STATE, probability=True)
    elif model_name == "XGBoost":
        base_model = xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            n_jobs=-1,
            verbosity=0
        )
    elif model_name == "Gradient Boosting":
        base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 创建 Pipeline
    pipeline = create_pipeline(base_model, use_scaled)
    
    print(f"  进行超参数搜索 (5折交叉验证)...")
    random_search = RandomizedSearchCV(
        pipeline, param_distributions, n_iter=n_iter, cv=cv, scoring='roc_auc', 
        n_jobs=-1, verbose=0, random_state=RANDOM_STATE
    )
    random_search.fit(X_train, y_train)
    
    print(f"  最佳参数: {random_search.best_params_}")
    print(f"  超参数搜索CV得分 (AUC): {random_search.best_score_:.4f}")
    
    best_pipeline = random_search.best_estimator_
    
    # 嵌套交叉验证评估
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1)
    nested_score, nested_std, nested_scores = nested_cv_evaluate(best_pipeline, X_train, y_train, cv_outer, model_name)
    
    print(f"\n  对最佳模型进行完整的5折交叉验证评估...")
    fold_results = cv_evaluate_model(best_pipeline, X_train, y_train, model_name, cv, use_proba=True)
    
    cv_results_summary = {
        'model': model_name,
        'cv_folds': 5,
        'accuracy_mean': np.mean(fold_results['accuracy']),
        'accuracy_std': np.std(fold_results['accuracy']),
        'precision_mean': np.mean(fold_results['precision']),
        'precision_std': np.std(fold_results['precision']),
        'recall_mean': np.mean(fold_results['recall']),
        'recall_std': np.std(fold_results['recall']),
        'f1_score_mean': np.mean(fold_results['f1_score']),
        'f1_score_std': np.std(fold_results['f1_score']),
        'roc_auc_mean': np.mean(fold_results['roc_auc']),
        'roc_auc_std': np.std(fold_results['roc_auc']),
        'balanced_accuracy_mean': np.mean(fold_results['balanced_accuracy']),
        'balanced_accuracy_std': np.std(fold_results['balanced_accuracy']),
        'mcc_mean': np.mean(fold_results['mcc']),
        'mcc_std': np.std(fold_results['mcc']),
        'pr_auc_mean': np.mean(fold_results['pr_auc']),
        'pr_auc_std': np.std(fold_results['pr_auc']),
        'nested_cv_score': nested_score,
        'nested_cv_std': nested_std
    }
    
    print(f"\n  5折交叉验证结果:")
    print(f"    Accuracy:        {cv_results_summary['accuracy_mean']:.4f} ± {cv_results_summary['accuracy_std']:.4f}")
    print(f"    Precision:       {cv_results_summary['precision_mean']:.4f} ± {cv_results_summary['precision_std']:.4f}")
    print(f"    Recall:          {cv_results_summary['recall_mean']:.4f} ± {cv_results_summary['recall_std']:.4f}")
    print(f"    F1-Score:        {cv_results_summary['f1_score_mean']:.4f} ± {cv_results_summary['f1_score_std']:.4f}")
    print(f"    ROC-AUC:         {cv_results_summary['roc_auc_mean']:.4f} ± {cv_results_summary['roc_auc_std']:.4f}")
    print(f"    PR-AUC:          {cv_results_summary['pr_auc_mean']:.4f} ± {cv_results_summary['pr_auc_std']:.4f}")
    print(f"    Balanced Acc:    {cv_results_summary['balanced_accuracy_mean']:.4f} ± {cv_results_summary['balanced_accuracy_std']:.4f}")
    print(f"    MCC:             {cv_results_summary['mcc_mean']:.4f} ± {cv_results_summary['mcc_std']:.4f}")
    print(f"    嵌套CV Score:    {cv_results_summary['nested_cv_score']:.4f} ± {cv_results_summary['nested_cv_std']:.4f}")
    
    print(f"\n  训练时间: {time.time() - start_time:.2f}秒")
    
    return best_pipeline, cv_results_summary, fold_results


def train_voting_ensemble_with_cv(X_train, y_train, base_pipelines):
    """训练投票集成模型并进行5折交叉验证"""
    print(f"\n{'='*70}")
    print(f"模型: Voting Ensemble (Soft Voting)")
    print("=" * 70)
    
    start_time = time.time()
    
    # 获取各基础模型的分类器
    estimators = []
    for name, pipeline in base_pipelines.items():
        if name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
            # 提取分类器
            if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
                clf = clone(pipeline.named_steps['classifier'])
                estimators.append((name.lower().replace(' ', '_'), clf))
    
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )
    
    # 创建 Pipeline
    voting_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', voting_clf)
    ])
    
    cv = get_cv_strategy()
    
    print(f"  对集成模型进行完整的5折交叉验证评估...")
    fold_results = cv_evaluate_model(voting_pipeline, X_train, y_train, "Voting Ensemble", cv, use_proba=True)
    
    voting_pipeline.fit(X_train, y_train)
    
    cv_results_summary = {
        'model': 'Voting Ensemble',
        'cv_folds': 5,
        'accuracy_mean': np.mean(fold_results['accuracy']),
        'accuracy_std': np.std(fold_results['accuracy']),
        'precision_mean': np.mean(fold_results['precision']),
        'precision_std': np.std(fold_results['precision']),
        'recall_mean': np.mean(fold_results['recall']),
        'recall_std': np.std(fold_results['recall']),
        'f1_score_mean': np.mean(fold_results['f1_score']),
        'f1_score_std': np.std(fold_results['f1_score']),
        'roc_auc_mean': np.mean(fold_results['roc_auc']),
        'roc_auc_std': np.std(fold_results['roc_auc']),
        'balanced_accuracy_mean': np.mean(fold_results['balanced_accuracy']),
        'balanced_accuracy_std': np.std(fold_results['balanced_accuracy']),
        'mcc_mean': np.mean(fold_results['mcc']),
        'mcc_std': np.std(fold_results['mcc']),
        'pr_auc_mean': np.mean(fold_results['pr_auc']),
        'pr_auc_std': np.std(fold_results['pr_auc']),
        'nested_cv_score': None,
        'nested_cv_std': None
    }
    
    print(f"\n  5折交叉验证结果:")
    print(f"    Accuracy:        {cv_results_summary['accuracy_mean']:.4f} ± {cv_results_summary['accuracy_std']:.4f}")
    print(f"    Precision:       {cv_results_summary['precision_mean']:.4f} ± {cv_results_summary['precision_std']:.4f}")
    print(f"    Recall:          {cv_results_summary['recall_mean']:.4f} ± {cv_results_summary['recall_std']:.4f}")
    print(f"    F1-Score:        {cv_results_summary['f1_score_mean']:.4f} ± {cv_results_summary['f1_score_std']:.4f}")
    print(f"    ROC-AUC:         {cv_results_summary['roc_auc_mean']:.4f} ± {cv_results_summary['roc_auc_std']:.4f}")
    print(f"    PR-AUC:          {cv_results_summary['pr_auc_mean']:.4f} ± {cv_results_summary['pr_auc_std']:.4f}")
    print(f"    Balanced Acc:    {cv_results_summary['balanced_accuracy_mean']:.4f} ± {cv_results_summary['balanced_accuracy_std']:.4f}")
    print(f"    MCC:             {cv_results_summary['mcc_mean']:.4f} ± {cv_results_summary['mcc_std']:.4f}")
    
    print(f"\n  训练时间: {time.time() - start_time:.2f}秒")
    
    return voting_pipeline, cv_results_summary, fold_results


def save_results(cv_results_all, fold_results_all, pipelines, X_train, y_train):
    """保存所有结果"""
    print("\n" + "=" * 70)
    print("保存结果")
    print("=" * 70)
    
    cv_summary_df = pd.DataFrame(cv_results_all)
    cv_summary_df.to_csv(f"{OUTPUT_DIR}/cv_results_summary.csv", index=False)
    print(f"✓ 交叉验证汇总结果已保存: {OUTPUT_DIR}/cv_results_summary.csv")
    
    fold_df = pd.DataFrame(fold_results_all)
    fold_df.to_csv(f"{OUTPUT_DIR}/cv_fold_results.csv", index=False)
    print(f"✓ 各折详细结果已保存: {OUTPUT_DIR}/cv_fold_results.csv")
    
    cv_roc_data = []
    cv_pr_data = []
    cv_confusion_matrix_data = []
    
    cv = get_cv_strategy()
    
    print("\n生成CV曲线数据...")
    
    for name, pipeline in pipelines.items():
        print(f"  处理 {name}...")
        
        X = X_train.values
        
        y_true_all = []
        y_proba_all = []
        
        for train_idx, val_idx in cv.split(X, y_train):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # 使用 clone() 克隆 Pipeline
            pipeline_clone = clone(pipeline)
            pipeline_clone.fit(X_fold_train, y_fold_train)
            
            y_proba = pipeline_clone.predict_proba(X_fold_val)[:, 1]
            y_true_all.extend(y_fold_val.values)
            y_proba_all.extend(y_proba)
        
        y_true_all = np.array(y_true_all)
        y_proba_all = np.array(y_proba_all)
        
        fpr, tpr, thresholds = roc_curve(y_true_all, y_proba_all)
        auc_score = roc_auc_score(y_true_all, y_proba_all)
        
        for i in range(len(fpr)):
            cv_roc_data.append({
                'Model': name,
                'FPR': fpr[i],
                'TPR': tpr[i],
                'Threshold': thresholds[i] if i < len(thresholds) else None,
                'AUC': auc_score
            })
        
        precision, recall, thresholds_pr = precision_recall_curve(y_true_all, y_proba_all)
        ap_score = average_precision_score(y_true_all, y_proba_all)
        
        for i in range(len(precision)):
            cv_pr_data.append({
                'Model': name,
                'Precision': precision[i],
                'Recall': recall[i],
                'Threshold': thresholds_pr[i] if i < len(thresholds_pr) else None,
                'AP': ap_score
            })
        
        y_pred_all = (y_proba_all >= 0.5).astype(int)
        cm = confusion_matrix(y_true_all, y_pred_all)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            cv_confusion_matrix_data.append({
                'Model': name,
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn),
                'TP': int(tp),
                'Threshold': 0.5
            })
    
    roc_df = pd.DataFrame(cv_roc_data)
    roc_df.to_csv(f"{OUTPUT_DIR}/cv_roc_curve_data.csv", index=False)
    print(f"✓ CV ROC曲线数据已保存: {OUTPUT_DIR}/cv_roc_curve_data.csv")
    
    pr_df = pd.DataFrame(cv_pr_data)
    pr_df.to_csv(f"{OUTPUT_DIR}/cv_precision_recall_curve_data.csv", index=False)
    print(f"✓ CV Precision-Recall曲线数据已保存: {OUTPUT_DIR}/cv_precision_recall_curve_data.csv")
    
    cm_df = pd.DataFrame(cv_confusion_matrix_data)
    cm_df.to_csv(f"{OUTPUT_DIR}/cv_confusion_matrix_data.csv", index=False)
    print(f"✓ CV 混淆矩阵数据已保存: {OUTPUT_DIR}/cv_confusion_matrix_data.csv")
    
    # 保存模型
    for name, pipeline in pipelines.items():
        model_path = f"{MODEL_DIR}/{name.lower().replace(' ', '_')}_cv_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"✓ 模型已保存: {model_path}")
    
    # 保存元数据
    metadata = {
        'script_version': 'optimized_v2',
        'random_state': RANDOM_STATE,
        'cv_folds': 5,
        'models': list(pipelines.keys()),
        'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'balanced_accuracy', 'mcc']
    }
    
    with open(f"{OUTPUT_DIR}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ 训练元数据已保存: {OUTPUT_DIR}/training_metadata.json")


def main():
    """主函数"""
    print("=" * 70)
    print("模型训练 - 5折交叉验证完整评估 (优化版)")
    print("=" * 70)
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    cv_results_all = []
    fold_results_all = []
    pipelines = {}
    
    # 超参数配置 - 使用 Pipeline 参数格式
    param_configs = {
        'Logistic Regression': {
            'params': {
                'classifier__C': uniform(0.001, 100),
                'classifier__penalty': ['l2'],
                'classifier__solver': ['lbfgs'],
                'classifier__class_weight': [None, 'balanced']
            },
            'n_iter': 30,
            'use_scaled': True
        },
        'Random Forest': {
            'params': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [5, 10, 15, 20, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__max_features': ['sqrt', 'log2'],
                'classifier__class_weight': [None, 'balanced']
            },
            'n_iter': 30,
            'use_scaled': False
        },
        'SVM': {
            'params': {
                'classifier__C': uniform(0.1, 100),
                'classifier__kernel': ['rbf', 'linear'],
                'classifier__gamma': ['scale', 'auto'],
                'classifier__class_weight': [None, 'balanced']
            },
            'n_iter': 20,
            'use_scaled': True
        },
        'XGBoost': {
            'params': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'classifier__subsample': [0.6, 0.8, 1.0],
                'classifier__colsample_bytree': [0.6, 0.8, 1.0],
                'classifier__min_child_weight': [1, 3, 5],
                'classifier__gamma': [0, 0.1, 0.2]
            },
            'n_iter': 30,
            'use_scaled': False
        },
        'Gradient Boosting': {
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.05, 0.1, 0.2],
                'classifier__subsample': [0.8, 1.0],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2]
            },
            'n_iter': 20,
            'use_scaled': False
        }
    }
    
    print("\n" + "=" * 70)
    print("开始训练各模型并进行5折交叉验证评估")
    print("=" * 70)
    
    for model_name, config in param_configs.items():
        best_pipeline, cv_summary, fold_results = train_and_evaluate_with_cv(
            X_train, y_train,
            model_name, config['params'], config['n_iter'], config['use_scaled']
        )
        
        cv_results_all.append(cv_summary)
        
        for fold_idx, fold_result in enumerate(fold_results['accuracy']):
            fold_results_all.append({
                'model': model_name,
                'fold': fold_idx + 1,
                'accuracy': fold_results['accuracy'][fold_idx],
                'precision': fold_results['precision'][fold_idx],
                'recall': fold_results['recall'][fold_idx],
                'f1_score': fold_results['f1_score'][fold_idx],
                'roc_auc': fold_results['roc_auc'][fold_idx],
                'pr_auc': fold_results['pr_auc'][fold_idx],
                'balanced_accuracy': fold_results['balanced_accuracy'][fold_idx],
                'mcc': fold_results['mcc'][fold_idx]
            })
        
        pipelines[model_name] = best_pipeline
    
    print("\n" + "=" * 70)
    print("训练投票集成模型")
    print("=" * 70)
    
    voting_pipeline, voting_cv_summary, voting_fold_results = train_voting_ensemble_with_cv(
        X_train, y_train, pipelines
    )
    
    cv_results_all.append(voting_cv_summary)
    
    for fold_idx in range(5):
        fold_results_all.append({
            'model': 'Voting Ensemble',
            'fold': fold_idx + 1,
            'accuracy': voting_fold_results['accuracy'][fold_idx],
            'precision': voting_fold_results['precision'][fold_idx],
            'recall': voting_fold_results['recall'][fold_idx],
            'f1_score': voting_fold_results['f1_score'][fold_idx],
            'roc_auc': voting_fold_results['roc_auc'][fold_idx],
            'pr_auc': voting_fold_results['pr_auc'][fold_idx],
            'balanced_accuracy': voting_fold_results['balanced_accuracy'][fold_idx],
            'mcc': voting_fold_results['mcc'][fold_idx]
        })
    
    pipelines['Voting Ensemble'] = voting_pipeline
    
    save_results(cv_results_all, fold_results_all, pipelines, X_train, y_train)
    
    print("\n" + "=" * 70)
    print("5折交叉验证结果汇总")
    print("=" * 70)
    
    print("\n{:<25} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1-Score", "PR-AUC"))
    print("-" * 85)
    
    for result in cv_results_all:
        print("{:<25} {:>6.3f}±{:<5.3f} {:>6.3f}±{:<5.3f} {:>6.3f}±{:<5.3f} {:>6.3f}±{:<5.3f} {:>6.3f}±{:<5.3f}".format(
            result['model'],
            result['accuracy_mean'], result['accuracy_std'],
            result['precision_mean'], result['precision_std'],
            result['recall_mean'], result['recall_std'],
            result['f1_score_mean'], result['f1_score_std'],
            result['pr_auc_mean'], result['pr_auc_std']
        ))
    
    print("\n{:<25} {:>15} {:>15} {:>15}".format(
        "Model", "ROC-AUC", "Balanced Acc", "MCC"))
    print("-" * 70)
    
    for result in cv_results_all:
        print("{:<25} {:>7.3f}±{:<6.3f} {:>7.3f}±{:<6.3f} {:>7.3f}±{:<6.3f}".format(
            result['model'],
            result['roc_auc_mean'], result['roc_auc_std'],
            result['balanced_accuracy_mean'], result['balanced_accuracy_std'],
            result['mcc_mean'], result['mcc_std']
        ))
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()