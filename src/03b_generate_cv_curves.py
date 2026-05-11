"""
CV曲线数据生成脚本
功能：从已训练的CV模型生成ROC曲线、PR曲线和混淆矩阵数据
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

TRAIN_PATH = "data/train_data_selected.csv"
OUTPUT_DIR = "results"
MODEL_DIR = "results/models"
RANDOM_STATE = 42


def load_data_and_models():
    """加载数据和模型"""
    print("=" * 60)
    print("加载数据和模型")
    print("=" * 60)
    
    train_data = pd.read_csv(TRAIN_PATH)
    y_train = train_data.iloc[:, -1]
    X_train_df = train_data.iloc[:, :-1]
    X_train = X_train_df.values
    
    print(f"训练集: {X_train.shape}")
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    
    models = {}
    model_names = ['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost', 'Gradient Boosting', 'Voting Ensemble']
    
    for name in model_names:
        model_path = f"{MODEL_DIR}/{name.lower().replace(' ', '_')}_cv_model.pkl"
        with open(model_path, 'rb') as f:
            models[name] = pickle.load(f)
        print(f"已加载模型: {name}")
    
    print(f"\n共加载 {len(models)} 个模型")
    
    return X_train, X_train_scaled, y_train, models


def get_cv_strategy():
    """获取5折分层交叉验证策略"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cv


def generate_cv_curve_data(X_train, X_train_scaled, y_train, models):
    """生成CV曲线数据"""
    print("\n" + "=" * 60)
    print("生成CV曲线数据")
    print("=" * 60)
    
    cv = get_cv_strategy()
    
    cv_roc_data = []
    cv_pr_data = []
    cv_confusion_matrix_data = []
    
    for name, model in models.items():
        print(f"\n处理 {name}...")
        
        use_scaled = True
        if name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
            use_scaled = False
        
        if use_scaled:
            X = X_train_scaled
        else:
            X = X_train
        
        y_true_all = []
        y_proba_all = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_train)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model_clone = pickle.loads(pickle.dumps(model))
            model_clone.fit(X_fold_train, y_fold_train)
            
            y_proba = model_clone.predict_proba(X_fold_val)[:, 1]
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
        tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
        
        cv_confusion_matrix_data.append({
            'Model': name,
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp),
            'Threshold': 0.5
        })
        
        print(f"  ROC-AUC: {auc_score:.4f}, AP: {ap_score:.4f}")
    
    return cv_roc_data, cv_pr_data, cv_confusion_matrix_data


def save_data(cv_roc_data, cv_pr_data, cv_confusion_matrix_data):
    """保存数据"""
    print("\n" + "=" * 60)
    print("保存数据")
    print("=" * 60)
    
    roc_df = pd.DataFrame(cv_roc_data)
    roc_df.to_csv(f"{OUTPUT_DIR}/cv_roc_curve_data.csv", index=False)
    print(f"CV ROC曲线数据已保存: {OUTPUT_DIR}/cv_roc_curve_data.csv")
    
    pr_df = pd.DataFrame(cv_pr_data)
    pr_df.to_csv(f"{OUTPUT_DIR}/cv_precision_recall_curve_data.csv", index=False)
    print(f"CV Precision-Recall曲线数据已保存: {OUTPUT_DIR}/cv_precision_recall_curve_data.csv")
    
    cm_df = pd.DataFrame(cv_confusion_matrix_data)
    cm_df.to_csv(f"{OUTPUT_DIR}/cv_confusion_matrix_data.csv", index=False)
    print(f"CV 混淆矩阵数据已保存: {OUTPUT_DIR}/cv_confusion_matrix_data.csv")
    
    print("\n混淆矩阵汇总:")
    print(cm_df.to_string(index=False))


def main():
    """主函数"""
    X_train, X_train_scaled, y_train, models = load_data_and_models()
    cv_roc_data, cv_pr_data, cv_confusion_matrix_data = generate_cv_curve_data(
        X_train, X_train_scaled, y_train, models
    )
    save_data(cv_roc_data, cv_pr_data, cv_confusion_matrix_data)
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
