"""
特征选择脚本
功能：对样本数据进行特征选择，使用多种方法筛选重要特征
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# 配置参数
SAMPLE_DIR = r"D:\trae\paper\input\features\test\featureselect\smalltest\sample"
OUTPUT_DIR = r"D:\trae\paper\input\features\test\featureselect\smalltest"
N_SPLITS = 5
N_REPEATS = 3
RANDOM_STATE = 42

def load_data():
    """加载数据"""
    print("=" * 60)
    print("Step 1: Load Data")
    print("=" * 60)
    
    X_all = np.load(os.path.join(SAMPLE_DIR, "X_all.npy"))
    y_all = np.load(os.path.join(SAMPLE_DIR, "y_all.npy"))
    ids_all = np.load(os.path.join(SAMPLE_DIR, "ids_all.npy"), allow_pickle=True)
    feature_names = np.load(os.path.join(SAMPLE_DIR, "feature_names.npy"), allow_pickle=True)
    test_idx = np.load(os.path.join(SAMPLE_DIR, "test_idx.npy"))
    
    print(f"X_all shape: {X_all.shape}")
    print(f"y_all shape: {y_all.shape}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Test set size: {len(test_idx)}")
    
    return X_all, y_all, ids_all, feature_names, test_idx

def variance_threshold_filter(X, y, threshold=0.1):
    """使用方差阈值过滤低方差特征"""
    print("\n" + "=" * 60)
    print("Step 2: Variance Threshold Filtering")
    print("=" * 60)
    
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    
    # 获取保留的特征索引
    selected_mask = selector.get_support()
    selected_indices = np.where(selected_mask)[0]
    
    print(f"Original features: {X.shape[1]}")
    print(f"After variance threshold (threshold={threshold}): {X_filtered.shape[1]}")
    print(f"Removed features: {X.shape[1] - X_filtered.shape[1]}")
    
    return X_filtered, selected_indices

def select_k_best(X, y, k=100):
    """使用SelectKBest选择最佳特征"""
    print("\n" + "=" * 60)
    print(f"Step 3: SelectKBest (k={k})")
    print("=" * 60)
    
    # 使用ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # 获取每个特征的分数
    scores = selector.scores_
    selected_indices = selector.get_support(indices=True)
    
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Top 10 feature scores:")
    top_indices = np.argsort(scores)[-10:][::-1]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Feature {idx}: score={scores[idx]:.4f}")
    
    return X_selected, selected_indices, scores

def feature_importance_selection(X, y, k=100):
    """使用随机森林特征重要性选择特征"""
    print("\n" + "=" * 60)
    print(f"Step 4: Random Forest Feature Importance (k={k})")
    print("=" * 60)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_scaled, y)
    
    # 获取特征重要性
    importances = rf.feature_importances_
    selected_indices = np.argsort(importances)[-k:]
    
    X_selected = X[:, selected_indices]
    
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Top 10 feature importances:")
    top_indices = np.argsort(importances)[-10:][::-1]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Feature {idx}: importance={importances[idx]:.4f}")
    
    return X_selected, selected_indices, importances

def combined_feature_selection(X, y, k=50):
    """组合多种方法进行特征选择"""
    print("\n" + "=" * 60)
    print(f"Step 5: Combined Feature Selection")
    print("=" * 60)
    
    # 方法1: ANOVA F-value
    selector1 = SelectKBest(score_func=f_classif, k=k*2)
    selector1.fit(X, y)
    scores_f = selector1.scores_
    selected_f = np.argsort(scores_f)[-k*2:]
    
    # 方法2: 互信息
    selector2 = SelectKBest(score_func=mutual_info_classif, k=k*2)
    selector2.fit(X, y)
    scores_mi = selector2.scores_
    selected_mi = np.argsort(scores_mi)[-k*2:]
    
    # 方法3: 随机森林
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_
    selected_rf = np.argsort(importances)[-k*2:]
    
    # 取交集
    from collections import Counter
    all_selected = np.concatenate([selected_f, selected_mi, selected_rf])
    counts = Counter(all_selected)
    # 选择在至少2种方法中被选中的特征
    selected_indices = np.array([idx for idx, count in counts.items() if count >= 2])
    
    # 如果太多，取前k个
    if len(selected_indices) > k:
        # 按平均排名选择
        rank_f = np.argsort(np.argsort(scores_f))
        rank_mi = np.argsort(np.argsort(scores_mi))
        rank_rf = np.argsort(np.argsort(importances))
        avg_rank = (rank_f + rank_mi + rank_rf) / 3
        selected_indices = np.argsort(avg_rank)[-k:]
    
    X_selected = X[:, selected_indices]
    
    print(f"Selected features: {X_selected.shape[1]}")
    
    return X_selected, selected_indices

def save_results(X_all, y_all, ids_all, feature_names, selected_indices, method_name):
    """保存特征选择结果"""
    print("\n" + "=" * 60)
    print(f"Step 6: Save Results ({method_name})")
    print("=" * 60)
    
    # 保存选择后的数据
    X_selected = X_all[:, selected_indices]
    selected_feature_names = feature_names[selected_indices]
    
    output_subdir = os.path.join(OUTPUT_DIR, method_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    # 保存特征数据
    np.save(os.path.join(output_subdir, "X_all.npy"), X_selected)
    np.save(os.path.join(output_subdir, "y_all.npy"), y_all)
    np.save(os.path.join(output_subdir, "ids_all.npy"), ids_all)
    np.save(os.path.join(output_subdir, "feature_names.npy"), selected_feature_names)
    np.save(os.path.join(output_subdir, "selected_indices.npy"), selected_indices)
    
    # 保存原始索引文件
    test_idx = np.load(os.path.join(SAMPLE_DIR, "test_idx.npy"))
    np.save(os.path.join(output_subdir, "test_idx.npy"), test_idx)
    
    # 保存交叉验证索引
    for repeat in range(1, N_REPEATS + 1):
        for fold in range(1, N_SPLITS + 1):
            train_idx = np.load(os.path.join(SAMPLE_DIR, f"cv_{repeat}_{fold}_train_idx.npy"))
            val_idx = np.load(os.path.join(SAMPLE_DIR, f"cv_{repeat}_{fold}_val_idx.npy"))
            np.save(os.path.join(output_subdir, f"cv_{repeat}_{fold}_train_idx.npy"), train_idx)
            np.save(os.path.join(output_subdir, f"cv_{repeat}_{fold}_val_idx.npy"), val_idx)
    
    print(f"Saved to: {output_subdir}")
    print(f"X_all shape: {X_selected.shape}")
    print(f"Feature names: {len(selected_feature_names)}")
    
    return output_subdir

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Feature Selection Pipeline")
    print("=" * 60)
    
    # 加载数据
    X_all, y_all, ids_all, feature_names, test_idx = load_data()
    
    # 尝试不同的特征数量
    k_values = [50, 100, 200]
    
    results = {}
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Testing k={k}")
        print(f"{'='*60}")
        
        # 使用随机森林特征重要性选择
        X_selected, selected_indices, importances = feature_importance_selection(X_all, y_all, k=k)
        
        method_name = f"rf_k{k}"
        output_dir = save_results(X_all, y_all, ids_all, feature_names, selected_indices, method_name)
        results[method_name] = output_dir
    
    print("\n" + "=" * 60)
    print("Feature Selection Complete!")
    print("=" * 60)
    print("\nGenerated directories:")
    for method, path in results.items():
        print(f"  - {method}: {path}")
    
    return results

if __name__ == "__main__":
    results = main()
