"""
多模型训练脚本
功能：使用多种分类模型进行训练和评估
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

# 配置参数
DATA_DIR = r"D:\trae\paper\input\features\test\featureselect\smalltest\rf_k100"
OUTPUT_DIR = r"D:\trae\paper\input\features\test\featureselect\smalltest"
N_SPLITS = 5
N_REPEATS = 3
RANDOM_STATE = 42

def load_data():
    """加载数据"""
    print("=" * 60)
    print("Step 1: Load Data")
    print("=" * 60)
    
    X_all = np.load(os.path.join(DATA_DIR, "X_all.npy"))
    y_all = np.load(os.path.join(DATA_DIR, "y_all.npy"))
    ids_all = np.load(os.path.join(DATA_DIR, "ids_all.npy"), allow_pickle=True)
    feature_names = np.load(os.path.join(DATA_DIR, "feature_names.npy"), allow_pickle=True)
    test_idx = np.load(os.path.join(DATA_DIR, "test_idx.npy"))
    
    print(f"X_all shape: {X_all.shape}")
    print(f"y_all shape: {y_all.shape}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Test set size: {len(test_idx)}")
    
    return X_all, y_all, ids_all, feature_names, test_idx

def get_models():
    """定义多个分类模型"""
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        ),
        
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE))
        ]),
        
        'SVM (Linear)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', probability=True, random_state=RANDOM_STATE))
        ]),
        
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5))
        ]),
        
        'Naive Bayes': GaussianNB(),
        
        'Decision Tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        ),
        
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_STATE))
        ])
    }
    
    return models

def evaluate_model(y_true, y_pred, y_prob=None):
    """评估模型性能"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }
    
    if y_prob is not None:
        try:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['ROC-AUC'] = 0.0
    
    return metrics

def train_and_evaluate():
    """训练和评估所有模型"""
    print("\n" + "=" * 60)
    print("Step 2: Train and Evaluate Models")
    print("=" * 60)
    
    # 加载数据
    X_all, y_all, ids_all, feature_names, test_idx = load_data()
    
    # 获取模型
    models = get_models()
    
    # 存储结果
    results = []
    cv_results = {name: {'accuracy': [], 'f1': [], 'roc_auc': []} for name in models.keys()}
    
    # 独立测试集
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]
    
    print(f"\nTest set: {len(y_test)} samples, class dist: {np.bincount(y_test)}")
    print("-" * 60)
    
    # 遍历每个模型
    for model_name, model in models.items():
        print(f"\n>>> {model_name}")
        
        # 存储测试集预测结果
        test_preds = []
        test_probs = []
        
        # 使用重复交叉验证
        for repeat in range(1, N_REPEATS + 1):
            for fold in range(1, N_SPLITS + 1):
                # 加载索引
                train_idx = np.load(os.path.join(DATA_DIR, f"cv_{repeat}_{fold}_train_idx.npy"))
                val_idx = np.load(os.path.join(DATA_DIR, f"cv_{repeat}_{fold}_val_idx.npy"))
                
                # 获取训练和验证数据
                X_train, y_train = X_all[train_idx], y_all[train_idx]
                X_val, y_val = X_all[val_idx], y_all[val_idx]
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 验证集预测
                y_pred = model.predict(X_val)
                try:
                    y_prob = model.predict_proba(X_val)[:, 1]
                except:
                    y_prob = None
                
                # 评估
                metrics = evaluate_model(y_val, y_pred, y_prob)
                cv_results[model_name]['accuracy'].append(metrics['Accuracy'])
                cv_results[model_name]['f1'].append(metrics['F1-Score'])
                cv_results[model_name]['roc_auc'].append(metrics['ROC-AUC'])
        
        # 在完整训练+验证集上训练，然后在测试集上评估
        train_val_idx = np.concatenate([np.load(os.path.join(DATA_DIR, f"cv_{repeat}_{fold}_train_idx.npy")) 
                                         for repeat in range(1, N_REPEATS + 1) 
                                         for fold in range(1, N_SPLITS + 1)])
        # 去重
        train_val_idx = np.unique(train_val_idx)
        
        X_train_val = X_all[train_val_idx]
        y_train_val = y_all[train_val_idx]
        
        # 重新创建模型实例
        model = get_models()[model_name]
        model.fit(X_train_val, y_train_val)
        
        # 测试集预测
        y_test_pred = model.predict(X_test)
        try:
            y_test_prob = model.predict_proba(X_test)[:, 1]
        except:
            y_test_prob = None
        
        # 测试集评估
        test_metrics = evaluate_model(y_test, y_test_pred, y_test_prob)
        
        # 计算CV平均值
        cv_acc_mean = np.mean(cv_results[model_name]['accuracy'])
        cv_acc_std = np.std(cv_results[model_name]['accuracy'])
        cv_f1_mean = np.mean(cv_results[model_name]['f1'])
        cv_f1_std = np.std(cv_results[model_name]['f1'])
        cv_auc_mean = np.mean(cv_results[model_name]['roc_auc'])
        cv_auc_std = np.std(cv_results[model_name]['roc_auc'])
        
        print(f"  CV Accuracy: {cv_acc_mean:.4f} (+/- {cv_acc_std:.4f})")
        print(f"  CV F1-Score: {cv_f1_mean:.4f} (+/- {cv_f1_std:.4f})")
        print(f"  CV ROC-AUC: {cv_auc_mean:.4f} (+/- {cv_auc_std:.4f})")
        print(f"  Test Accuracy: {test_metrics['Accuracy']:.4f}")
        print(f"  Test F1-Score: {test_metrics['F1-Score']:.4f}")
        print(f"  Test ROC-AUC: {test_metrics['ROC-AUC']:.4f}")
        
        results.append({
            'Model': model_name,
            'CV Accuracy': f"{cv_acc_mean:.4f} (+/- {cv_acc_std:.4f})",
            'CV F1': f"{cv_f1_mean:.4f} (+/- {cv_f1_std:.4f})",
            'CV ROC-AUC': f"{cv_auc_mean:.4f} (+/- {cv_auc_std:.4f})",
            'Test Accuracy': f"{test_metrics['Accuracy']:.4f}",
            'Test F1': f"{test_metrics['F1-Score']:.4f}",
            'Test ROC-AUC': f"{test_metrics['ROC-AUC']:.4f}",
            'cv_acc_mean': cv_acc_mean,
            'cv_f1_mean': cv_f1_mean,
            'cv_auc_mean': cv_auc_mean
        })
    
    return results

def save_results(results):
    """保存结果"""
    print("\n" + "=" * 60)
    print("Step 3: Save Results")
    print("=" * 60)
    
    # 创建DataFrame
    df_results = pd.DataFrame(results)
    
    # 按CV F1分数排序
    df_results = df_results.sort_values('cv_f1_mean', ascending=False)
    
    # 保存CSV
    output_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    df_results.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # 打印排名
    print("\n" + "=" * 60)
    print("Model Ranking (by CV F1-Score)")
    print("=" * 60)
    print(df_results[['Model', 'CV F1', 'CV ROC-AUC', 'Test F1', 'Test ROC-AUC']].to_string(index=False))
    
    return df_results

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Multi-Model Training Pipeline")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"CV splits: {N_SPLITS}")
    print(f"CV repeats: {N_REPEATS}")
    print(f"Total evaluations: {N_SPLITS * N_REPEATS}")
    
    # 训练和评估
    results = train_and_evaluate()
    
    # 保存结果
    df_results = save_results(results)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return df_results

if __name__ == "__main__":
    df_results = main()
