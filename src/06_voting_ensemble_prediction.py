"""
Voting Ensemble 模型预测脚本
功能：使用训练好的 Voting Ensemble 模型对新数据进行预测
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# 导入必要的模块 (确保模型依赖可用)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 路径配置
MODEL_PATH = "results/models/voting_ensemble_cv_model.pkl"
FEATURES_PATH = "data/features_processed.csv"
SELECTED_FEATURES_PATH = "data/selected_features.csv"
OUTPUT_PATH = "results/predictions.csv"

def load_model(model_path):
    """
    加载训练好的模型
    """
    print("=" * 70)
    print("加载模型")
    print("=" * 70)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"模型加载成功: {model_path}")
    print(f"模型类型: {type(model)}")
    
    return model


def load_selected_features(selected_features_path):
    """
    加载选中的特征列表
    """
    print("\n" + "=" * 70)
    print("加载选中特征")
    print("=" * 70)
    
    if not os.path.exists(selected_features_path):
        raise FileNotFoundError(f"特征文件不存在: {selected_features_path}")
    
    selected_features_df = pd.read_csv(selected_features_path)
    selected_features = selected_features_df['feature'].tolist()
    
    print(f"选中特征数量: {len(selected_features)}")
    print(f"前5个特征: {selected_features[:5]}")
    
    return selected_features


def load_features_data(features_path, selected_features):
    """
    加载特征数据并筛选选中特征
    """
    print("\n" + "=" * 70)
    print("加载特征数据")
    print("=" * 70)
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"特征数据文件不存在: {features_path}")
    
    # 加载特征数据
    features_df = pd.read_csv(features_path)
    
    print(f"原始数据形状: {features_df.shape}")
    print(f"原始列名: {features_df.columns[:5].tolist()} ...")
    
    # 检查数据是否有标签列 (最后一列)
    # 假设最后一列是标签
    last_col = features_df.columns[-1]
    print(f"最后一列名称: {last_col}")
    
    # 判断最后一列是否是标签 (检查是否为数值类型且唯一值较少)
    is_label = False
    if features_df[last_col].dtype in ['int64', 'int32', 'float64', 'float32']:
        unique_vals = features_df[last_col].nunique()
        if unique_vals <= 10:  # 假设标签类别不超过10个
            is_label = True
            print(f"检测到标签列: {last_col}, 类别数: {unique_vals}")
    
    if is_label:
        # 分离特征和标签
        y = features_df[last_col]
        X = features_df.drop(columns=[last_col])
    else:
        # 没有标签列，全部作为特征
        X = features_df
        y = None
        print("未检测到标签列，将对所有数据进行预测")
    
    # 筛选选中特征
    available_features = [f for f in selected_features if f in X.columns]
    missing_features = [f for f in selected_features if f not in X.columns]
    
    if missing_features:
        print(f"警告: 有 {len(missing_features)} 个选中特征在数据中不存在")
        print(f"缺失特征示例: {missing_features[:5]}")
    
    print(f"可用特征数量: {len(available_features)}")
    
    X_selected = X[available_features]
    
    print(f"筛选后数据形状: {X_selected.shape}")
    
    return X_selected, y


def predict(model, X):
    """
    使用模型进行预测
    """
    print("\n" + "=" * 70)
    print("进行预测")
    print("=" * 70)
    
    # 进行预测
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    print(f"预测完成")
    print(f"预测结果分布: {np.unique(y_pred, return_counts=True)}")
    print(f"预测概率形状: {y_pred_proba.shape}")
    
    return y_pred, y_pred_proba


def save_predictions(X, y, y_pred, y_pred_proba, output_path):
    """
    保存预测结果
    """
    print("\n" + "=" * 70)
    print("保存预测结果")
    print("=" * 70)
    
    # 创建结果 DataFrame
    results = pd.DataFrame()
    
    # 添加原始特征 (前10个)
    feature_cols = X.columns[:10].tolist()
    for col in feature_cols:
        results[col] = X[col].values
    
    # 如果有真实标签，添加
    if y is not None:
        results['true_label'] = y.values
    
    # 添加预测结果
    results['predicted_label'] = y_pred
    
    # 添加预测概率
    results['probability_class_0'] = y_pred_proba[:, 0]
    results['probability_class_1'] = y_pred_proba[:, 1]
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存结果
    results.to_csv(output_path, index=False)
    
    print(f"预测结果已保存至: {output_path}")
    print(f"结果数据形状: {results.shape}")
    
    # 输出预测统计
    if y is not None:
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y, y_pred)
        print(f"\n预测准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y, y_pred))
    
    return results


def main():
    """
    主函数
    """
    print("\n" + "=" * 70)
    print("Voting Ensemble 模型预测")
    print("=" * 70)
    
    # 1. 加载模型
    model = load_model(MODEL_PATH)
    
    # 2. 加载选中特征列表
    selected_features = load_selected_features(SELECTED_FEATURES_PATH)
    
    # 3. 加载特征数据并筛选
    X, y = load_features_data(FEATURES_PATH, selected_features)
    
    # 4. 进行预测
    y_pred, y_pred_proba = predict(model, X)
    
    # 5. 保存预测结果
    results = save_predictions(X, y, y_pred, y_pred_proba, OUTPUT_PATH)
    
    print("\n" + "=" * 70)
    print("预测完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()