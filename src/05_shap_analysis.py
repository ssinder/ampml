#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
SHAP Model Interpretability Analysis Script
功能：仅进行SHAP分析和保存中间数据，不生成图表

Features:
1. SHAP analysis for Voting Ensemble base models
2. Compute SHAP values with optimized performance
3. Save intermediate data for publication-quality figures

Optimized based on SHAP SKILL best practices:
- Auto-detect optimal Explainer type
- Performance optimization for large datasets
- Robust error handling with fallback
'''

import pandas as pd
import numpy as np
import pickle
import os
import warnings
import json
import time
from datetime import datetime

import shap

warnings.filterwarnings('ignore')

# 路径配置
DATA_DIR = 'data'
MODEL_DIR = 'results/models'
OUTPUT_DIR = 'results/shap'
RANDOM_STATE = 42

# 性能优化配置
MAX_SAMPLES_FOR_SHAP = 500  # 大数据集采样限制
MAX_SAMPLES_FOR_SVM = 50   # SVM特别限制（KernelExplainer很慢）
BATCH_SIZE = 100  # 批量处理大小

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=' * 70)
print('SHAP Model Interpretability Analysis')
print('=' * 70)
print(f'Analysis Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print('=' * 70)


def load_data_and_models():
    '''
    加载数据和各单模型
    
    Returns:
        X_train: 训练特征
        X_test: 测试特征
        y_train: 训练标签
        y_test: 测试标签
        models: 预训练的模型字典
        feature_names: 特征名称列表
    '''
    print('\n[1] Loading data...')
    
    # 加载测试数据
    test_data = pd.read_csv(f'{DATA_DIR}/test_data_selected.csv')
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    
    # 加载训练数据（用于SHAP background）
    train_data = pd.read_csv(f'{DATA_DIR}/train_data_selected.csv')
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    
    # 加载选中的特征列表
    selected_features = pd.read_csv(f'{DATA_DIR}/selected_features.csv')
    feature_names = selected_features['feature'].tolist()
    
    print(f'  Test set shape: {X_test.shape}')
    print(f'  Train set shape: {X_train.shape}')
    print(f'  Number of selected features: {len(feature_names)}')
    
    # 加载各单模型
    print('\n[2] Loading models...')
    models = {}
    
    model_files = {
        'Logistic Regression': 'logistic_regression_cv_model.pkl',
        'Random Forest': 'random_forest_cv_model.pkl',
        'XGBoost': 'xgboost_cv_model.pkl',
        'Gradient Boosting': 'gradient_boosting_cv_model.pkl',
        'SVM': 'svm_cv_model.pkl'
    }
    
    for model_name, filename in model_files.items():
        filepath = f'{MODEL_DIR}/{filename}'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                models[model_name] = pickle.load(f)
            print(f'  Loaded: {model_name}')
        else:
            print(f'  Warning: Model file not found - {filename}')
    
    return X_train, X_test, y_train, y_test, models, feature_names


def compute_shap_values(model, X_train, X_test, model_name):
    '''
    计算模型的SHAP值
    
    使用shap.Explainer自动检测最佳解释器（SKILL最佳实践）
    性能优化：大数据集采样 + 批量处理
    
    Args:
        model: 训练好的模型
        X_train: 训练数据（用于background）
        X_test: 测试数据
        model_name: 模型名称
    
    Returns:
        explainer: SHAP解释器对象
        shap_values: SHAP值矩阵
        X_test_sample: 实际用于计算SHAP的测试样本
        base_value: 基础值（期望输出）
    '''
    print(f'\n  Computing SHAP values for {model_name}...')
    start_time = time.time()
    
    # 提取分类器
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        classifier = model.named_steps['classifier']
    else:
        classifier = model
    
    # 性能优化：对大数据集进行采样
    n_train = len(X_train)
    n_test = len(X_test)
    
    # 采样训练数据用于background
    if n_train > MAX_SAMPLES_FOR_SHAP:
        sample_indices = np.random.choice(n_train, MAX_SAMPLES_FOR_SHAP, replace=False)
        X_train_sample = X_train.iloc[sample_indices]
    else:
        X_train_sample = X_train
    
    # 采样测试数据
    # SVM使用KernelExplainer非常慢，需要更严格的限制
    max_samples = MAX_SAMPLES_FOR_SVM if 'SVM' in str(type(classifier)) else MAX_SAMPLES_FOR_SHAP
    if n_test > max_samples:
        sample_indices = np.random.choice(n_test, max_samples, replace=False)
        X_test_sample = X_test.iloc[sample_indices]
    else:
        X_test_sample = X_test
    
    print(f'    Using {len(X_train_sample)} background samples, {len(X_test_sample)} test samples')
    
    try:
        # 使用shap.Explainer自动检测最佳解释器（SKILL推荐）
        explainer = shap.Explainer(classifier, X_train_sample)
        
        # 计算SHAP值
        shap_result = explainer(X_test_sample)
        
        # 处理不同输出格式
        if hasattr(shap_result, 'values'):
            shap_values = shap_result.values
        else:
            shap_values = np.array(shap_result)
        
        # 处理二分类：取正类SHAP值
        if len(shap_values.shape) == 3:
            # 多输出情况，取正类
            shap_values = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[:, :, 0]
        elif len(shap_values.shape) == 2 and shap_values.shape[1] == 2:
            shap_values = shap_values[:, 1]
        
        # 确保是二维数组
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
        
        elapsed = time.time() - start_time
        print(f'    Completed! Time: {elapsed:.2f}s')
        
        # 获取base_value
        base_value = shap_result.base_values if hasattr(shap_result, 'base_values') else None
        if base_value is not None and len(base_value.shape) > 0:
            base_value = base_value[0] if hasattr(base_value, '__iter__') else base_value
        
        return explainer, shap_values, X_test_sample, base_value
        
    except Exception as e:
        print(f'    Error: {e}')
        print('    Trying fallback with KernelExplainer...')
        # SVM使用更小的样本数
        max_s = MAX_SAMPLES_FOR_SVM if 'SVM' in str(type(classifier)) else 50
        return compute_shap_values_fallback(classifier, X_train_sample, X_test_sample, start_time, max_s)


def compute_shap_values_fallback(classifier, X_train, X_test, start_time, max_samples=50):
    '''
    备用SHAP计算方法 - 使用KernelExplainer
    
    作为最后手段的回退方案
    
    Args:
        classifier: 分类器对象
        X_train: 训练数据
        X_test: 测试数据
        start_time: 开始时间
        max_samples: 最大样本数
    
    Returns:
        explainer: SHAP解释器对象
        shap_values: SHAP值矩阵
        X_test_sample: 实际用于计算SHAP的测试样本
        base_value: 基础值
    '''
    try:
        # 采样测试数据
        if len(X_test) > max_samples:
            sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_test = X_test.iloc[sample_indices]
        
        # 使用代表性样本减少计算时间
        X_train_kmeans = shap.kmeans(X_train, min(50, len(X_train)))
        explainer = shap.KernelExplainer(classifier.predict_proba, X_train_kmeans)
        
        # 批量处理SHAP值计算
        all_shap_values = []
        n_samples = len(X_test)
        
        for i in range(0, n_samples, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, n_samples)
            batch = X_test.iloc[i:batch_end]
            batch_shap = explainer.shap_values(batch)
            
            if isinstance(batch_shap, list):
                batch_shap = batch_shap[1]  # 取正类
            all_shap_values.append(batch_shap)
        
        shap_values = np.vstack(all_shap_values)
        
        elapsed = time.time() - start_time
        print(f'    Completed with KernelExplainer! Time: {elapsed:.2f}s')
        
        return explainer, shap_values, X_test, None
        
    except Exception as e:
        print(f'    Fallback also failed: {e}')
        return None, None, X_test, None


def save_shap_summary(all_shap_results, output_path):
    '''
    保存SHAP分析汇总结果
    
    Args:
        all_shap_results: 所有模型的SHAP结果字典
        output_path: 输出文件路径
    '''
    print(f'\n  Saving SHAP summary...')
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    summary = {}
    for model_name, result in all_shap_results.items():
        mean_abs_shap = np.abs(result['shap_values']).mean(axis=0)
        
        # 确保是一维数组
        mean_abs_shap = np.asarray(mean_abs_shap).flatten()
        
        # 获取排序后的索引
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        
        # 确保feature_names是列表
        feature_names = list(result['feature_names'])
        n_features = len(feature_names)
        
        top_features = []
        for i in range(min(20, len(sorted_idx))):
            idx = sorted_idx[i]
            if idx < n_features:
                top_features.append({
                    'feature': feature_names[idx],
                    'mean_abs_shap': float(mean_abs_shap[idx]),
                    'std_abs_shap': float(np.std(result['shap_values'][:, idx])),
                    'rank': i + 1
                })
        
        summary[model_name] = {
            'top_20_features': top_features,
            'mean_global_importance': float(mean_abs_shap.mean()),
            'std_global_importance': float(mean_abs_shap.std()),
            'max_importance': float(mean_abs_shap.max()),
            'min_importance': float(mean_abs_shap.min()),
            'n_samples': result['shap_values'].shape[0],
            'n_features': result['shap_values'].shape[1]
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f'    Saved: {output_path}')
    
    # 打印Top10特征
    print('\n  Top 10 Features by SHAP Importance:')
    print('-' * 70)
    for model_name, data in summary.items():
        print(f'\n  {model_name}:')
        for feat in data['top_20_features'][:10]:
            rank = feat['rank']
            feature = feat['feature']
            mean_abs = feat['mean_abs_shap']
            print(f'    {rank:2d}. {feature:<50} {mean_abs:.4f}')


def save_intermediate_data(all_shap_results, X_test, feature_names, output_path):
    '''
    保存SHAP中间数据用于后续顶刊级别图表制作
    
    保存内容：
    1. 每个模型的SHAP值矩阵
    2. 特征重要性排名表
    3. 测试数据特征矩阵
    4. 特征名称列表
    5. 模型间特征重要性相关性矩阵
    6. 模型比较汇总（Top特征交集分析）
    
    Args:
        all_shap_results: 所有模型的SHAP结果字典
        X_test: 测试数据
        feature_names: 特征名称列表
        output_path: 输出目录路径
    '''
    print(f'\n  Saving SHAP intermediate data...')
    
    # 1. 保存每个模型的SHAP值矩阵
    shap_values_dir = f'{output_path}/shap_values'
    os.makedirs(shap_values_dir, exist_ok=True)
    
    for model_name, result in all_shap_results.items():
        safe_name = model_name.lower().replace(' ', '_')
        shap_values = result['shap_values']
        
        # 处理可能的3D数组（多输出情况）
        if len(shap_values.shape) == 3:
            # 取正类
            shap_values = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[:, :, 0]
        elif len(shap_values.shape) == 2 and shap_values.shape[1] == 2:
            shap_values = shap_values[:, 1]
        
        # 确保是2D数组
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
        
        # 保存为CSV
        shap_df = pd.DataFrame(
            shap_values,
            columns=feature_names[:shap_values.shape[1]] if shap_values.shape[1] <= len(feature_names) else feature_names
        )
        shap_df.to_csv(f'{shap_values_dir}/{safe_name}_shap_values.csv', index=False)
        print(f'    Saved: {shap_values_dir}/{safe_name}_shap_values.csv')
    
    # 2. 保存特征重要性汇总表（所有模型）
    importance_data = []
    feature_names_list = list(feature_names)
    
    for model_name, result in all_shap_results.items():
        mean_abs_shap = np.abs(result['shap_values']).mean(axis=0)
        mean_abs_shap = np.asarray(mean_abs_shap).flatten()
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        
        for rank in range(min(len(sorted_idx), len(feature_names_list))):
            idx = sorted_idx[rank]
            if idx < len(feature_names_list):
                importance_data.append({
                    'model': model_name,
                    'rank': rank + 1,
                    'feature': feature_names_list[idx],
                    'mean_abs_shap': float(mean_abs_shap[idx]),
                    'std_abs_shap': float(np.abs(result['shap_values'])[:, idx].std()) if result['shap_values'].shape[0] > 1 else 0.0
                })
    
    importance_df = pd.DataFrame(importance_data)
    importance_df.to_csv(f'{output_path}/feature_importance_ranking.csv', index=False)
    print(f'    Saved: {output_path}/feature_importance_ranking.csv')
    
    # 3. 保存测试数据特征矩阵
    X_test.to_csv(f'{output_path}/test_features.csv', index=False)
    print(f'    Saved: {output_path}/test_features.csv')
    
    # 4. 保存特征名称列表
    feature_df = pd.DataFrame({'feature': feature_names})
    feature_df.to_csv(f'{output_path}/feature_names.csv', index=False)
    print(f'    Saved: {output_path}/feature_names.csv')
    
    # 5. 保存模型间特征重要性相关性矩阵
    model_names_list = list(all_shap_results.keys())
    corr_matrix = []
    
    for result1 in all_shap_results.values():
        importance1 = np.abs(result1['shap_values']).mean(axis=0)
        importance1 = np.asarray(importance1).flatten()  # 确保是1D
        row = []
        for result2 in all_shap_results.values():
            importance2 = np.abs(result2['shap_values']).mean(axis=0)
            importance2 = np.asarray(importance2).flatten()  # 确保是1D
            min_len = min(len(importance1), len(importance2))
            corr = np.corrcoef(importance1[:min_len], importance2[:min_len])[0, 1]
            row.append(corr)
        corr_matrix.append(row)
    
    corr_df = pd.DataFrame(corr_matrix, index=model_names_list, columns=model_names_list)
    corr_df.to_csv(f'{output_path}/model_correlation_matrix.csv')
    print(f'    Saved: {output_path}/model_correlation_matrix.csv')
    
    # 6. 保存模型比较汇总（Top特征交集分析）
    top_features_sets = {}
    for model_name, result in all_shap_results.items():
        mean_abs_shap = np.abs(result['shap_values']).mean(axis=0)
        mean_abs_shap = np.asarray(mean_abs_shap).flatten()
        top_indices = np.argsort(mean_abs_shap)[:20]
        top_features_sets[model_name] = set([feature_names[i] for i in top_indices if i < len(feature_names)])
    
    # 找出所有模型共同的重要特征
    common_features = set.intersection(*top_features_sets.values()) if top_features_sets else set()
    
    # 找出每个模型独有的重要特征
    unique_features = {}
    for model_name, features_set in top_features_sets.items():
        other_features = set.union(*[s for m, s in top_features_sets.items() if m != model_name])
        unique_features[model_name] = features_set - other_features
    
    comparison_summary = {
        'common_top20_features': list(common_features),
        'unique_top20_features': {k: list(v) for k, v in unique_features.items()},
        'n_models': len(all_shap_results),
        'n_common_features': len(common_features)
    }
    
    with open(f'{output_path}/model_comparison_summary.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_summary, f, indent=2, ensure_ascii=False)
    print(f'    Saved: {output_path}/model_comparison_summary.json')
    
    # 7. 保存完整的SHAP结果（包含base_value等）
    full_results = {}
    for model_name, result in all_shap_results.items():
        base_val = result['base_value']
        if base_val is not None:
            base_val = float(np.asarray(base_val).flatten()[0]) if hasattr(base_val, '__len__') else float(base_val)
        
        full_results[model_name] = {
            'shap_values_shape': result['shap_values'].shape,
            'feature_names': result['feature_names'],
            'base_value': base_val,
            'X_test_sample_info': {
                'n_samples': len(result['X_test']),
                'columns': list(result['X_test'].columns)
            }
        }
    
    with open(f'{output_path}/shap_results_info.json', 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f'    Saved: {output_path}/shap_results_info.json')
    
    print(f'\n  Intermediate data saved! Data for {len(all_shap_results)} models')
    print(f'  Output directory: {output_path}')


def main():
    '''
    主函数：执行SHAP分析和数据保存
    '''
    total_start_time = time.time()
    
    # 加载数据和模型
    X_train, X_test, _, _, models, feature_names = load_data_and_models()
    
    if not models:
        print('Error: No model files found!')
        return
    
    # 存储所有SHAP结果
    all_shap_results = {}
    
    # 对每个模型计算SHAP值
    print('\n[3] Computing SHAP values for each model...')
    for model_name, model in models.items():
        # 跳过SVM模型（KernelExplainer太慢）
        if 'SVM' in model_name:
            print(f'  Skipping {model_name} (too slow with KernelExplainer)')
            continue
        
        try:
            _, shap_values, X_test_sample, base_value = compute_shap_values(
                model, X_train, X_test, model_name
            )
            
            if shap_values is not None:
                all_shap_results[model_name] = {
                    'shap_values': shap_values,
                    'X_test': X_test_sample,
                    'feature_names': feature_names,
                    'base_value': base_value
                }
                print(f'  Successfully computed SHAP for {model_name}')
            else:
                print(f'  Failed to compute SHAP for {model_name}')
                
        except Exception as e:
            print(f'  Error processing {model_name}: {e}')
            continue
    
    if not all_shap_results:
        print('Error: No SHAP values computed successfully!')
        return
    
    print(f'\n  Successfully processed {len(all_shap_results)} models')
    
    # 保存SHAP汇总结果
    print('\n[4] Saving SHAP summary...')
    save_shap_summary(all_shap_results, f'{OUTPUT_DIR}/shap_summary.json')
    
    # 保存中间数据用于顶刊做图
    print('\n[5] Saving intermediate data for publication...')
    save_intermediate_data(all_shap_results, X_test, feature_names, OUTPUT_DIR)
    
    # 打印完成信息
    total_elapsed = time.time() - total_start_time
    print('\n' + '=' * 70)
    print('SHAP Analysis Completed!')
    print(f'Total Time: {total_elapsed:.2f} seconds')
    print(f'Output Directory: {OUTPUT_DIR}')
    print('=' * 70)


if __name__ == '__main__':
    main()
