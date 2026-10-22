"""
样本提取与划分脚本
功能：从原始数据中提取少量样本，划分独立测试集，使用重复分层交叉验证划分训练集/验证集
简化版：只保存索引文件，减少冗余存储
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
import os

# 配置参数
DATA_DIR = r"D:\trae\paper\input\features\test\featureselect"
OUTPUT_DIR = r"D:\trae\paper\input\features\test\featureselect\smalltest"
SAMPLE_SIZE = 200  # 少量样本测试
TEST_SIZE = 20  # 独立测试集大小
N_SPLITS = 5  # 每折数
N_REPEATS = 3  # 重复次数
RANDOM_STATE = 42

def load_data():
    """加载特征和标签数据"""
    print("=" * 60)
    print("Step 1: Load Data")
    print("=" * 60)
    
    features_path = os.path.join(DATA_DIR, "features_processed.csv")
    labels_path = os.path.join(DATA_DIR, "labels.csv")
    
    print(f"Loading features: {features_path}")
    df_features = pd.read_csv(features_path)
    print(f"Features shape: {df_features.shape}")
    
    print(f"\nLoading labels: {labels_path}")
    df_labels = pd.read_csv(labels_path)
    print(f"Labels shape: {df_labels.shape}")
    
    return df_features, df_labels

def extract_samples(df_features, df_labels, sample_size):
    """提取少量样本用于测试"""
    print("\n" + "=" * 60)
    print("Step 2: Extract Samples")
    print("=" * 60)
    
    # 基于ID合并特征和标签
    df_merged = pd.merge(df_features, df_labels, on='ID', how='inner')
    print(f"Merged data shape: {df_merged.shape}")
    
    # 检查类别分布
    label_counts = df_merged['Label'].value_counts()
    print(f"\nOriginal class distribution:")
    print(label_counts)
    
    # 分层抽样以保持类别比例
    samples_per_class = sample_size // 2
    
    df_sample_list = []
    for label in df_merged['Label'].unique():
        df_class = df_merged[df_merged['Label'] == label]
        n_samples = min(samples_per_class, len(df_class))
        df_sample_list.append(df_class.sample(n=n_samples, random_state=RANDOM_STATE))
    
    df_sample = pd.concat(df_sample_list, ignore_index=True)
    df_sample = df_sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    print(f"\nExtracted samples: {len(df_sample)}")
    print(f"Sample class distribution:")
    print(df_sample['Label'].value_counts())
    
    return df_sample

def split_data(df_sample, test_size=20):
    """划分独立测试集和交叉验证数据集"""
    print("\n" + "=" * 60)
    print("Step 3: Split Independent Test Set")
    print("=" * 60)
    
    # 分离特征和标签
    X = df_sample.drop(['ID', 'Label'], axis=1).values
    y = df_sample['Label'].values
    ids = df_sample['ID'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")
    
    # 划分独立测试集（分层抽样）
    train_val_idx, test_idx = train_test_split(
        np.arange(len(X)), 
        test_size=test_size, 
        stratify=y, 
        random_state=RANDOM_STATE
    )
    
    print(f"\nIndependent test set split:")
    print(f"  Test set: {len(test_idx)} samples, class dist: {np.bincount(y[test_idx])}")
    print(f"  Train+Val set: {len(train_val_idx)} samples, class dist: {np.bincount(y[train_val_idx])}")
    
    return X, y, ids, train_val_idx, test_idx

def cross_validation_split(train_val_idx, y, n_splits=5, n_repeats=3):
    """使用重复分层交叉验证划分训练集/验证集"""
    print("\n" + "=" * 60)
    print(f"Step 4: {n_splits}-Fold Repeated Stratified CV Split ({n_repeats} repeats)")
    print("=" * 60)
    
    # 创建重复分层K折划分器
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats, 
        random_state=RANDOM_STATE
    )
    
    # 存储每一折的索引
    cv_splits = []
    
    y_train_val = y[train_val_idx]
    
    print(f"\n{n_splits}-Fold CV x {n_repeats} repeats = {n_splits * n_repeats} total evaluations")
    print("-" * 60)
    
    total_folds = n_splits * n_repeats
    for repeat_idx in range(n_repeats):
        for fold_idx in range(n_splits):
            # 获取当前折的索引
            current_fold = repeat_idx * n_splits + fold_idx
            
            # 获取训练和验证索引
            train_idx_local = list(rskf.split(train_val_idx, y_train_val))[current_fold][0]
            val_idx_local = list(rskf.split(train_val_idx, y_train_val))[current_fold][1]
            
            # 将索引转换回原始数据的索引
            train_idx_original = train_val_idx[train_idx_local]
            val_idx_original = train_val_idx[val_idx_local]
            
            split_info = {
                'repeat': repeat_idx + 1,
                'fold': fold_idx + 1,
                'total_fold': current_fold + 1,
                'train_idx': train_idx_original,
                'val_idx': val_idx_original
            }
            
            cv_splits.append(split_info)
            
            if fold_idx == 0:
                print(f"\nRepeat {repeat_idx + 1}:")
            print(f"  Fold {fold_idx + 1}: Train={len(train_idx_original)}, Val={len(val_idx_original)}, "
                  f"Class dist=[{np.bincount(y[train_idx_original])} / {np.bincount(y[val_idx_original])}]")
    
    print("\n" + "=" * 60)
    print("Split complete!")
    print("=" * 60)
    
    return cv_splits

def save_splits(X, y, ids, test_idx, cv_splits, df_sample):
    """保存划分结果（简化版：只保存索引）"""
    print("\n" + "=" * 60)
    print("Step 5: Save Split Results (Simplified)")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取特征名称
    feature_names = df_sample.drop(['ID', 'Label'], axis=1).columns.tolist()
    
    # ===== 核心数据文件 =====
    # 完整数据（通过索引获取）
    np.save(os.path.join(OUTPUT_DIR, "X_all.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_all.npy"), y)
    np.save(os.path.join(OUTPUT_DIR, "ids_all.npy"), ids)
    print(f"Saved: X_all={X.shape}, y_all={y.shape}")
    
    # 特征名称
    np.save(os.path.join(OUTPUT_DIR, "feature_names.npy"), np.array(feature_names))
    print(f"Saved: feature_names ({len(feature_names)} features)")
    
    # 完整样本数据（CSV，便于查看）
    df_sample.to_csv(os.path.join(OUTPUT_DIR, "sample_data.csv"), index=False)
    print(f"Saved: sample_data.csv")
    
    # ===== 索引文件 =====
    # 独立测试集索引（所有折共用）
    np.save(os.path.join(OUTPUT_DIR, "test_idx.npy"), test_idx)
    print(f"Saved: test_idx (size={len(test_idx)})")
    
    # 每折的训练/验证索引
    for split_info in cv_splits:
        repeat = split_info['repeat']
        fold = split_info['fold']
        # 使用 repeat_fold 格式命名，如 1_1, 1_2, ..., 3_5
        np.save(os.path.join(OUTPUT_DIR, f"cv_{repeat}_{fold}_train_idx.npy"), split_info['train_idx'])
        np.save(os.path.join(OUTPUT_DIR, f"cv_{repeat}_{fold}_val_idx.npy"), split_info['val_idx'])
    
    # 打印摘要
    n_repeats = max([s['repeat'] for s in cv_splits])
    n_folds = max([s['fold'] for s in cv_splits])
    print(f"\nTotal CV splits: {len(cv_splits)} ({n_repeats} repeats x {n_folds} folds)")
    print(f"Example: cv_1_1_train_idx.npy, cv_1_1_val_idx.npy, ...")
    
    # ===== 划分摘要 =====
    summary = []
    summary.append("=" * 60)
    summary.append("Sample Split Summary")
    summary.append("=" * 60)
    summary.append(f"Total Samples: {len(df_sample)}")
    summary.append(f"Feature Count: {len(feature_names)}")
    summary.append(f"CV Folds: {N_SPLITS}")
    summary.append(f"CV Repeats: {N_REPEATS}")
    summary.append(f"Total Evaluations: {N_SPLITS * N_REPEATS}")
    summary.append(f"Random Seed: {RANDOM_STATE}")
    summary.append("")
    summary.append("Test Set (Independent, shared by all folds):")
    summary.append(f"  Size: {len(test_idx)}")
    summary.append(f"  Class distribution: {np.bincount(y[test_idx])}")
    summary.append("")
    summary.append("Per-fold data (first repeat):")
    first_repeat_splits = [s for s in cv_splits if s['repeat'] == 1]
    for split_info in first_repeat_splits:
        fold = split_info['fold']
        summary.append(f"  Fold {fold}: Train={len(split_info['train_idx'])}, Val={len(split_info['val_idx'])}, Test={len(test_idx)}")
    
    with open(os.path.join(OUTPUT_DIR, "split_summary.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    print(f"\nSaved: split_summary.txt")
    
    print("\n" + "=" * 60)
    print("All files saved!")
    print("=" * 60)

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Sample Extraction and Split with Repeated CV")
    print("=" * 60)
    print(f"Config:")
    print(f"  - Sample size: {SAMPLE_SIZE}")
    print(f"  - Test set size: {TEST_SIZE}")
    print(f"  - CV folds: {N_SPLITS}")
    print(f"  - CV repeats: {N_REPEATS}")
    print(f"  - Total evaluations: {N_SPLITS * N_REPEATS}")
    print(f"  - Random seed: {RANDOM_STATE}")
    print(f"  - Output dir: {OUTPUT_DIR}")
    
    # 加载数据
    df_features, df_labels = load_data()
    
    # 提取样本
    df_sample = extract_samples(df_features, df_labels, SAMPLE_SIZE)
    
    # 划分独立测试集（返回索引）
    X, y, ids, train_val_idx, test_idx = split_data(df_sample, TEST_SIZE)
    
    # 重复交叉验证划分
    cv_splits = cross_validation_split(train_val_idx, y, N_SPLITS, N_REPEATS)
    
    # 保存结果
    save_splits(X, y, ids, test_idx, cv_splits, df_sample)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return cv_splits, df_sample

if __name__ == "__main__":
    cv_splits, df_sample = main()
