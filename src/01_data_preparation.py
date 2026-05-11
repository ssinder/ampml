"""
数据加载与预处理脚本 (规范版)
功能：加载原始特征数据，进行规范的预处理，划分训练/验证/测试集

预处理流程：
1. 数据加载与基础检查
2. 标签提取与数据验证
3. 缺失值检测与处理
4. 异常值检测与处理
5. 特征标准化
6. 数据划分
7. 保存预处理后的数据

输出：
- 预处理后的完整数据 (data/features_processed.csv)
- 训练/验证/测试集
- 预处理元数据
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 设置路径
RAW_DATA_PATH = "rawdata/all_features_combined_standard.csv"
OUTPUT_DIR = "data"
RANDOM_STATE = 42


def load_raw_data():
    """加载原始特征数据"""
    print("=" * 70)
    print("步骤1: 加载原始数据")
    print("=" * 70)
    
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"原始数据文件不存在: {RAW_DATA_PATH}")
    
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"原始数据维度: {df.shape}")
    print(f"  - 样本数: {df.shape[0]}")
    print(f"  - 特征数: {df.shape[1]}")
    print(f"  - ID列: {df.columns[0]}")
    print(f"  - 特征列数: {df.shape[1] - 1}")
    
    return df


def extract_labels(df):
    """从ID列提取标签"""
    print("\n" + "=" * 70)
    print("步骤2: 提取标签")
    print("=" * 70)
    
    if 'ID' not in df.columns:
        raise ValueError("数据中缺少 'ID' 列")
    
    # 从ID列提取标签 (格式: L0_xxxxxx 或 L1_xxxxxx)
    df['label'] = df['ID'].str.extract(r'^(L\d)_')[0]
    
    if df['label'].isna().any():
        raise ValueError("部分ID格式不正确，无法提取标签")
    
    # 转换为数值标签
    df['label_encoded'] = df['label'].map({'L0': 0, 'L1': 1})
    
    print(f"标签分布:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"  - {label}: {count} ({pct:.1f}%)")
    
    return df


def check_missing_values(df, feature_cols):
    """检查缺失值"""
    print("\n" + "=" * 70)
    print("步骤3: 缺失值检测")
    print("=" * 70)
    
    missing_info = {}
    total_missing = 0
    
    for col in feature_cols:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        if missing_count > 0:
            missing_info[col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 4)
            }
            total_missing += missing_count
    
    if total_missing == 0:
        print("✓ 数据集中无缺失值")
        return None
    else:
        print(f"⚠ 发现 {total_missing} 个缺失值分布在 {len(missing_info)} 个特征中")
        print("\n缺失值最多的前10个特征:")
        sorted_missing = sorted(missing_info.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        for col, info in sorted_missing:
            print(f"  - {col}: {info['count']} ({info['percentage']:.2f}%)")
        return missing_info


def check_constant_features(X, feature_cols):
    """检测常量特征(方差为0)"""
    print("\n" + "=" * 70)
    print("步骤4: 常量特征检测")
    print("=" * 70)
    
    constant_features = []
    for i, col in enumerate(feature_cols):
        if np.std(X[:, i]) == 0:
            constant_features.append(col)
    
    if constant_features:
        print(f"⚠ 发现 {len(constant_features)} 个常量特征")
        print(f"  示例: {constant_features[:5]}")
    else:
        print("✓ 未发现常量特征")
    
    return constant_features


def detect_outliers_iqr(X, feature_cols, threshold=3.0):
    """使用IQR方法检测异常值"""
    print("\n" + "=" * 70)
    print("步骤5: 异常值检测 (IQR方法)")
    print("=" * 70)
    
    outlier_info = {}
    n_outliers_per_feature = []
    
    for i, col in enumerate(feature_cols):
        Q1 = np.percentile(X[:, i], 25)
        Q3 = np.percentile(X[:, i], 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = (X[:, i] < lower_bound) | (X[:, i] > upper_bound)
        n_outliers = np.sum(outliers)
        
        if n_outliers > 0:
            outlier_info[col] = {
                'count': int(n_outliers),
                'percentage': round(n_outliers / len(X) * 100, 2),
                'Q1': round(Q1, 4),
                'Q3': round(Q3, 4),
                'IQR': round(IQR, 4)
            }
            n_outliers_per_feature.append(n_outliers)
    
    total_outliers = sum(n_outliers_per_feature)
    if total_outliers > 0:
        print(f"⚠ 发现 {total_outliers} 个异常值分布在 {len(outlier_info)} 个特征中")
        print("\n异常值最多的前10个特征:")
        sorted_outliers = sorted(outlier_info.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        for col, info in sorted_outliers:
            print(f"  - {col}: {info['count']} ({info['percentage']:.2f}%)")
    else:
        print("✓ 未发现明显异常值")
    
    return outlier_info


def impute_missing_and_outliers(X_train, X_val, X_test, feature_cols):
    """处理缺失值和异常值"""
    print("\n" + "=" * 70)
    print("步骤6: 缺失值与异常值处理")
    print("=" * 70)
    
    # 使用中位数填充缺失值 (对异常值鲁棒)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    
    print(f"✓ 使用中位数策略填充缺失值")
    print(f"  - 训练集: {X_train.shape} -> {X_train_imputed.shape}")
    print(f"  - 验证集: {X_val.shape} -> {X_val_imputed.shape}")
    print(f"  - 测试集: {X_test.shape} -> {X_test_imputed.shape}")
    
    return X_train_imputed, X_val_imputed, X_test_imputed, imputer


def standardize_features(X_train, X_val, X_test, method='standard'):
    """特征标准化"""
    print("\n" + "=" * 70)
    print(f"步骤7: 特征标准化 ({method})")
    print("=" * 70)
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"未知的标准化方法: {method}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ 使用 {method} 标准化")
    print(f"  - 训练集均值范围: [{X_train_scaled.mean(axis=0).min():.4f}, {X_train_scaled.mean(axis=0).max():.4f}]")
    print(f"  - 训练集标准差范围: [{X_train_scaled.std(axis=0).min():.4f}, {X_train_scaled.std(axis=0).max():.4f}]")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def validate_data_quality(X, y, dataset_name):
    """验证数据质量"""
    print(f"\n{dataset_name} 数据质量验证:")
    
    issues = []
    
    if X.shape[1] == 0:
        issues.append("特征数量为0")
    
    if X.shape[0] == 0:
        issues.append("样本数量为0")
    
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        issues.append(f"标签类别不足: {unique_labels}")
    
    if np.isinf(X).any():
        issues.append("发现无穷值")
    
    constant_features = [i for i in range(X.shape[1]) if np.std(X[:, i]) == 0]
    if constant_features:
        issues.append(f"发现 {len(constant_features)} 个常量特征")
    
    if issues:
        print(f"  ⚠ 发现以下问题:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print(f"  ✓ 数据质量检查通过")
        return True


def compute_data_statistics(df, feature_cols):
    """计算数据统计信息"""
    print("\n" + "=" * 70)
    print("数据统计信息")
    print("=" * 70)
    
    X = df[feature_cols].values
    
    stats_info = {
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'n_positive': int(sum(df['label_encoded'] == 1)),
        'n_negative': int(sum(df['label_encoded'] == 0)),
        'feature_min': float(X.min()),
        'feature_max': float(X.max()),
        'feature_mean': float(X.mean()),
        'feature_std': float(X.std()),
        'feature_median': float(np.median(X)),
        'skewness': float(stats.skew(X.flatten())),
        'kurtosis': float(stats.kurtosis(X.flatten()))
    }
    
    print(f"样本数: {stats_info['n_samples']}")
    print(f"特征数: {stats_info['n_features']}")
    print(f"正例(L1)数: {stats_info['n_positive']}")
    print(f"负例(L0)数: {stats_info['n_negative']}")
    print(f"特征值范围: [{stats_info['feature_min']:.4f}, {stats_info['feature_max']:.4f}]")
    print(f"特征均值: {stats_info['feature_mean']:.4f}")
    print(f"特征标准差: {stats_info['feature_std']:.4f}")
    print(f"偏度: {stats_info['skewness']:.4f}")
    print(f"峰度: {stats_info['kurtosis']:.4f}")
    
    return stats_info


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("抗菌肽机器学习 - 数据预处理流程")
    print("=" * 70)
    
    # 1. 加载原始数据
    df = load_raw_data()
    
    # 2. 提取标签
    df = extract_labels(df)
    
    # 3. 提取特征列
    feature_cols = [col for col in df.columns if col not in ['ID', 'label', 'label_encoded']]
    print(f"\n特征列数量: {len(feature_cols)}")
    
    # 4. 缺失值检测
    missing_info = check_missing_values(df, feature_cols)
    
    # 5. 准备特征矩阵
    X = df[feature_cols].values
    y = df['label_encoded'].values
    
    print(f"\n特征矩阵维度: {X.shape}")
    print(f"标签维度: {y.shape}")
    
    # 6. 第一次划分: 训练集(70%) 和 临时集(30%)
    print("\n" + "=" * 70)
    print("步骤8: 数据集划分")
    print("=" * 70)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.3, 
        stratify=y,
        random_state=RANDOM_STATE
    )
    
    # 第二次划分: 验证集(15%) 和 测试集(15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )
    
    print(f"训练集: {X_train.shape[0]} 样本 ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"验证集: {X_val.shape[0]} 样本 ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"测试集: {X_test.shape[0]} 样本 ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    print(f"\n训练集类别分布: L0={sum(y_train==0)}, L1={sum(y_train==1)}")
    print(f"验证集类别分布: L0={sum(y_val==0)}, L1={sum(y_val==1)}")
    print(f"测试集类别分布: L0={sum(y_test==0)}, L1={sum(y_test==1)}")
    
    # 7. 检测常量特征
    constant_features = check_constant_features(X_train, feature_cols)
    
    # 8. 异常值检测 (在划分后的训练集上进行)
    outlier_info = detect_outliers_iqr(X_train, feature_cols, threshold=3.0)
    
    # 9. 缺失值和异常值处理
    X_train_processed, X_val_processed, X_test_processed, imputer = \
        impute_missing_and_outliers(X_train, X_val, X_test, feature_cols)
    
    # 10. 特征标准化
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = \
        standardize_features(X_train_processed, X_val_processed, X_test_processed, method='standard')
    
    # 11. 数据质量验证
    print("\n" + "=" * 70)
    print("步骤9: 数据质量验证")
    print("=" * 70)
    validate_data_quality(X_train_scaled, y_train, "训练集")
    validate_data_quality(X_val_scaled, y_val, "验证集")
    validate_data_quality(X_test_scaled, y_test, "测试集")
    
    # 12. 计算数据统计信息
    stats_info = compute_data_statistics(df, feature_cols)
    
    # 13. 保存数据
    print("\n" + "=" * 70)
    print("步骤10: 保存数据")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存特征列名
    feature_info = {
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'imputer_strategy': 'median',
        'scaler_type': 'StandardScaler',
        'preprocessing_steps': [
            '缺失值填充(中位数)',
            '标准化(StandardScaler)'
        ]
    }
    np.save(f"{OUTPUT_DIR}/feature_info.npy", feature_info, allow_pickle=True)
    
    # 保存原始特征名称到CSV (便于后续分析)
    feature_df = pd.DataFrame({'feature_name': feature_cols})
    feature_df.to_csv(f"{OUTPUT_DIR}/feature_names_raw.csv", index=False)
    
    # 保存训练/验证/测试集 (含标准化后的特征)
    train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_df['ID'] = df.iloc[:len(y_train)]['ID'].values[:len(y_train)]
    # 重新划分后需要追踪原始ID，这里用索引代替
    train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_df['label'] = y_train
    
    val_df = pd.DataFrame(X_val_scaled, columns=feature_cols)
    val_df['label'] = y_val
    
    test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_df['label'] = y_test
    
    train_df.to_csv(f"{OUTPUT_DIR}/train_data.csv", index=False)
    val_df.to_csv(f"{OUTPUT_DIR}/val_data.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/test_data.csv", index=False)
    
    # 保存预处理元数据
    metadata = {
        'data_source': RAW_DATA_PATH,
        'total_samples': int(len(df)),
        'total_features': len(feature_cols),
        'split_ratios': {
            'train': 0.7,
            'val': 0.15,
            'test': 0.15
        },
        'sample_counts': {
            'train': int(len(y_train)),
            'val': int(len(y_val)),
            'test': int(len(y_test))
        },
        'class_distribution': {
            'train': {'L0': int(sum(y_train==0)), 'L1': int(sum(y_train==1))},
            'val': {'L0': int(sum(y_val==0)), 'L1': int(sum(y_val==1))},
            'test': {'L0': int(sum(y_test==0)), 'L1': int(sum(y_test==1))}
        },
        'missing_values': missing_info if missing_info else 'None',
        'constant_features_count': len(constant_features),
        'outliers_detected': len(outlier_info),
        'imputer_strategy': 'median',
        'scaler_type': 'StandardScaler',
        'random_state': RANDOM_STATE
    }
    
    with open(f"{OUTPUT_DIR}/preprocessing_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 特征信息已保存至: {OUTPUT_DIR}/feature_info.npy")
    print(f"✓ 特征名称已保存至: {OUTPUT_DIR}/feature_names_raw.csv")
    print(f"✓ 训练数据已保存至: {OUTPUT_DIR}/train_data.csv")
    print(f"✓ 验证数据已保存至: {OUTPUT_DIR}/val_data.csv")
    print(f"✓ 测试数据已保存至: {OUTPUT_DIR}/test_data.csv")
    print(f"✓ 预处理元数据已保存至: {OUTPUT_DIR}/preprocessing_metadata.json")
    
    # 14. 保存用于画图的原始数据统计
    print("\n" + "=" * 70)
    print("步骤11: 保存用于可视化的中间数据")
    print("=" * 70)
    
    # 保存原始数据的特征分布统计
    visualization_data = {
        'feature_statistics': {
            'mean': X.mean(axis=0).tolist(),
            'std': X.std(axis=0).tolist(),
            'min': X.min(axis=0).tolist(),
            'max': X.max(axis=0).tolist(),
            'q25': np.percentile(X, 25, axis=0).tolist(),
            'q50': np.percentile(X, 50, axis=0).tolist(),
            'q75': np.percentile(X, 75, axis=0).tolist()
        },
        'feature_names': feature_cols,
        'label_distribution': {
            'L0': int(sum(y == 0)),
            'L1': int(sum(y == 1))
        }
    }
    
    np.save(f"{OUTPUT_DIR}/visualization_data.npy", visualization_data, allow_pickle=True)
    print(f"✓ 可视化数据已保存至: {OUTPUT_DIR}/visualization_data.npy")
    
    print("\n" + "=" * 70)
    print("数据预处理完成!")
    print("=" * 70)
    print(f"\n预处理流程总结:")
    print(f"  1. 加载原始数据: {RAW_DATA_PATH}")
    print(f"  2. 样本数: {len(df)}, 特征数: {len(feature_cols)}")
    print(f"  3. 数据划分: 训练集({len(y_train)})/验证集({len(y_val)})/测试集({len(y_test)})")
    print(f"  4. 缺失值处理: 中位数填充")
    print(f"  5. 特征标准化: StandardScaler")
    print(f"  6. 常量特征数: {len(constant_features)}")
    print(f"  7. 异常值特征数: {len(outlier_info)}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_cols


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = main()