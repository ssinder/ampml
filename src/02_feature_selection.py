"""
特征选择脚本 (增强优化版)
功能：对高维特征进行选择，降低维度同时保留重要特征

优化内容：
1. 添加更多特征选择方法（互信息、RFE、稳定性选择）
2. 自动选择最佳特征选择方法
3. 使用更稳健的评估策略
4. 保存完整的特征选择元数据
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold, SelectFromModel, SelectKBest, f_classif,
    mutual_info_classif, RFE, RFECV, SelectPercentile
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
import json
import time
warnings.filterwarnings('ignore')

# 设置路径
TRAIN_PATH = "data/train_data.csv"
VAL_PATH = "data/val_data.csv"
TEST_PATH = "data/test_data.csv"
OUTPUT_DIR = "data"
RANDOM_STATE = 42


def load_data():
    """加载数据"""
    print("=" * 70)
    print("加载数据")
    print("=" * 70)
    
    train_data = pd.read_csv(TRAIN_PATH)
    val_data = pd.read_csv(VAL_PATH)
    test_data = pd.read_csv(TEST_PATH)
    
    # 分离特征和标签
    y_train = train_data.iloc[:, -1]
    y_val = val_data.iloc[:, -1]
    y_test = test_data.iloc[:, -1]
    
    X_train = train_data.iloc[:, :-1]
    X_val = val_data.iloc[:, :-1]
    X_test = test_data.iloc[:, :-1]
    
    print(f"原始特征维度: {X_train.shape[1]}")
    print(f"训练集样本数: {X_train.shape[0]}")
    print(f"验证集样本数: {X_val.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_feature_selection(X_train, y_train, feature_selector, method_name, cv_folds=5):
    """使用交叉验证评估特征选择效果"""
    print(f"\n  评估 {method_name} 特征选择效果 ({cv_folds}-fold CV)...")
    
    pipeline = Pipeline([
        ('selector', feature_selector),
        ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs'))
    ])
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    try:
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"    CV ROC-AUC: {mean_score:.4f} ± {std_score:.4f}")
        return mean_score, std_score
    except Exception as e:
        print(f"    评估失败: {e}")
        return None, None


def variance_threshold_selection(X_train, X_val, X_test, threshold=0.0):
    """基于方差阈值的特征选择 - 移除常量特征"""
    print("\n" + "=" * 70)
    print("方法1: 方差阈值选择 (移除常量特征)")
    print("=" * 70)
    
    selector = VarianceThreshold(threshold=threshold)
    X_train_sel = selector.fit_transform(X_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"方差阈值: {threshold}")
    print(f"选择后特征数: {X_train_sel.shape[1]}")
    print(f"移除特征数: {X_train.shape[1] - X_train_sel.shape[1]}")
    
    return X_train_sel, X_val_sel, X_test_sel, selected_features, selector


def l1_logistic_selection_cv(X_train, X_val, X_test, y_train, max_features=300, C_values=[0.1, 0.5, 1.0]):
    """基于L1正则化逻辑回归的特征选择 - 自动选择最佳C值"""
    print("\n" + "=" * 70)
    print("方法2: L1正则化逻辑回归特征选择")
    print("=" * 70)
    
    best_score = 0
    best_C = 0.5
    best_selector = None
    
    # 测试不同的C值
    for C in C_values:
        print(f"\n  测试 C={C}...")
        selector = SelectFromModel(
            LogisticRegression(
                penalty='l1',
                solver='saga',
                C=C,
                max_iter=2000,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            threshold='median',
            max_features=max_features
        )
        
        cv_score, cv_std = evaluate_feature_selection(
            X_train, y_train, selector, f"L1 (C={C})"
        )
        
        if cv_score and cv_score > best_score:
            best_score = cv_score
            best_C = C
            best_selector = selector
    
    print(f"\n  最佳C值: {best_C}, CV ROC-AUC: {best_score:.4f}")
    
    # 使用最佳C值重新训练
    selector = SelectFromModel(
        LogisticRegression(
            penalty='l1',
            solver='saga',
            C=best_C,
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        threshold='median',
        max_features=max_features
    )
    
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    # 如果特征不足，补充
    if len(selected_features) < max_features:
        lr = LogisticRegression(penalty='l1', solver='saga', C=best_C, max_iter=2000, 
                                random_state=RANDOM_STATE, n_jobs=-1)
        lr.fit(X_train, y_train)
        
        importance = np.abs(lr.coef_[0])
        remaining_features = [col for col in X_train.columns if col not in selected_features]
        remaining_importance = [(col, importance[X_train.columns.get_loc(col)]) for col in remaining_features]
        remaining_importance.sort(key=lambda x: x[1], reverse=True)
        
        needed = max_features - len(selected_features)
        additional_features = [f[0] for f in remaining_importance[:needed]]
        
        all_features = selected_features + additional_features
        X_train_sel = X_train[all_features]
        X_val_sel = X_val[all_features]
        X_test_sel = X_test[all_features]
        selected_features = all_features
    else:
        X_train_sel = selector.transform(X_train)
        X_val_sel = selector.transform(X_val)
        X_test_sel = selector.transform(X_test)
    
    print(f"L1正则化逻辑回归选择特征数: {len(selected_features)}")
    
    # 重新评估最终结果
    final_selector = SelectFromModel(
        LogisticRegression(penalty='l1', solver='saga', C=best_C, max_iter=2000, 
                           random_state=RANDOM_STATE, n_jobs=-1),
        threshold='median'
    )
    cv_score, cv_std = evaluate_feature_selection(X_train, y_train, final_selector, "L1 Final")
    
    return X_train_sel, X_val_sel, X_test_sel, selected_features, best_C, (cv_score, cv_std)


def rf_importance_selection_cv(X_train, X_val, X_test, y_train, max_features=300, n_estimators_list=[100, 200]):
    """基于随机森林特征重要性的特征选择 - 自动选择最佳参数"""
    print("\n" + "=" * 70)
    print("方法3: 随机森林特征重要性选择")
    print("=" * 70)
    
    best_score = 0
    best_n = 100
    best_selector = None
    
    for n_estimators in n_estimators_list:
        print(f"\n  测试 n_estimators={n_estimators}...")
        selector = SelectFromModel(
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            threshold='median',
            max_features=max_features
        )
        
        cv_score, cv_std = evaluate_feature_selection(
            X_train, y_train, selector, f"RF (n={n_estimators})"
        )
        
        if cv_score and cv_score > best_score:
            best_score = cv_score
            best_n = n_estimators
            best_selector = selector
    
    print(f"\n  最佳n_estimators: {best_n}, CV ROC-AUC: {best_score:.4f}")
    
    # 使用最佳参数
    selector = SelectFromModel(
        RandomForestClassifier(
            n_estimators=best_n,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        threshold='median',
        max_features=max_features
    )
    
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    X_train_sel = selector.transform(X_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    
    print(f"随机森林特征重要性选择: {len(selected_features)} 个特征")
    
    # 重新评估
    final_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=best_n, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        threshold='median'
    )
    cv_score, cv_std = evaluate_feature_selection(X_train, y_train, final_selector, "RF Final")
    
    return X_train_sel, X_val_sel, X_test_sel, selected_features, best_n, (cv_score, cv_std)


def mutual_info_selection_cv(X_train, X_val, X_test, y_train, k_values=[200, 250, 300]):
    """基于互信息的特征选择"""
    print("\n" + "=" * 70)
    print("方法4: 互信息特征选择")
    print("=" * 70)
    
    best_score = 0
    best_k = 150
    best_selector = None
    
    for k in k_values:
        print(f"\n  测试 k={k}...")
        selector = SelectKBest(mutual_info_classif, k=k)
        
        cv_score, cv_std = evaluate_feature_selection(
            X_train, y_train, selector, f"MI (k={k})"
        )
        
        if cv_score and cv_score > best_score:
            best_score = cv_score
            best_k = k
            best_selector = selector
    
    print(f"\n  最佳k: {best_k}, CV ROC-AUC: {best_score:.4f}")
    
    selector = SelectKBest(mutual_info_classif, k=best_k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"互信息选择特征数: {len(selected_features)}")
    
    cv_score, cv_std = evaluate_feature_selection(X_train, y_train, selector, "MI Final")
    
    return X_train_sel, X_val_sel, X_test_sel, selected_features, best_k, (cv_score, cv_std)


def selectkbest_selection_cv(X_train, X_val, X_test, y_train, k_values=[200, 250, 300]):
    """使用SelectKBest (ANOVA F-test) 进行特征选择"""
    print("\n" + "=" * 70)
    print("方法5: SelectKBest (ANOVA F-test)")
    print("=" * 70)
    
    best_score = 0
    best_k = 150
    best_selector = None
    
    for k in k_values:
        print(f"\n  测试 k={k}...")
        selector = SelectKBest(f_classif, k=k)
        
        cv_score, cv_std = evaluate_feature_selection(
            X_train, y_train, selector, f"F-test (k={k})"
        )
        
        if cv_score and cv_score > best_score:
            best_score = cv_score
            best_k = k
            best_selector = selector
    
    print(f"\n  最佳k: {best_k}, CV ROC-AUC: {best_score:.4f}")
    
    selector = SelectKBest(f_classif, k=best_k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"SelectKBest选择特征数: {len(selected_features)}")
    
    cv_score, cv_std = evaluate_feature_selection(X_train, y_train, selector, "F-test Final")
    
    return X_train_sel, X_val_sel, X_test_sel, selected_features, best_k, (cv_score, cv_std)


def rfecv_selection(X_train, X_val, X_test, y_train, n_features_to_select=100):
    """使用RFECV (递归特征消除与交叉验证) 进行特征选择"""
    print("\n" + "=" * 70)
    print("方法6: RFECV (递归特征消除与交叉验证)")
    print("=" * 70)
    
    base_estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs')
    
    rfecv = RFECV(
        estimator=base_estimator,
        step=20,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring='roc_auc',
        min_features_to_select=n_features_to_select,
        n_jobs=-1
    )
    
    print(f"\n  正在进行RFECV (这可能需要一些时间)...")
    rfecv.fit(X_train, y_train)
    
    selected_features = X_train.columns[rfecv.support_].tolist()
    
    X_train_sel = rfecv.transform(X_train)
    X_val_sel = rfecv.transform(X_val)
    X_test_sel = rfecv.transform(X_test)
    
    print(f"RFECV选择特征数: {len(selected_features)}")
    print(f"最佳特征数: {rfecv.n_features_}")
    
    cv_score = rfecv.cv_results_['mean_test_score'].max()
    cv_std = rfecv.cv_results_['std_test_score'][rfecv.cv_results_['mean_test_score'].argmax()]
    print(f"  CV ROC-AUC: {cv_score:.4f} ± {cv_std:.4f}")
    
    return X_train_sel, X_val_sel, X_test_sel, selected_features, cv_score, (cv_score, cv_std)


def select_percentile_selection(X_train, X_val, X_test, y_train, percentile_values=[5, 10, 15, 20]):
    """使用SelectPercentile进行特征选择"""
    print("\n" + "=" * 70)
    print("方法6: SelectPercentile (基于F-test)")
    print("=" * 70)
    
    best_score = 0
    best_pct = 10
    best_selector = None
    
    for pct in percentile_values:
        print(f"\n  测试 percentile={pct}%...")
        selector = SelectPercentile(f_classif, percentile=pct)
        
        cv_score, cv_std = evaluate_feature_selection(
            X_train, y_train, selector, f"Percentile ({pct}%)"
        )
        
        if cv_score and cv_score > best_score:
            best_score = cv_score
            best_pct = pct
            best_selector = selector
    
    print(f"\n  最佳percentile: {best_pct}%, CV ROC-AUC: {best_score:.4f}")
    
    selector = SelectPercentile(f_classif, percentile=best_pct)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"SelectPercentile选择特征数: {len(selected_features)}")
    
    cv_score, cv_std = evaluate_feature_selection(X_train, y_train, selector, "Percentile Final")
    
    return X_train_sel, X_val_sel, X_test_sel, selected_features, best_pct, (cv_score, cv_std)


def main():
    """主函数"""
    print("=" * 70)
    print("特征选择流程 (增强优化版)")
    print("=" * 70)
    
    start_time = time.time()
    
    # 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # 存储不同方法的结果
    results = {}
    evaluation_results = {}
    
    # 方法1: L1正则化逻辑回归
    X_train_l1, X_val_l1, X_test_l1, features_l1, best_C_l1, cv_l1 = l1_logistic_selection_cv(
        X_train, X_val, X_test, y_train, max_features=300, C_values=[0.1, 0.5, 1.0]
    )
    results['L1_Logistic'] = {
        'features': features_l1,
        'X_train': X_train_l1,
        'X_val': X_val_l1,
        'X_test': X_test_l1,
        'best_param': f'C={best_C_l1}'
    }
    evaluation_results['L1_Logistic'] = {'cv_score': cv_l1[0], 'cv_std': cv_l1[1], 'n_features': len(features_l1)}
    
    # 方法2: 随机森林特征重要性
    X_train_rf, X_val_rf, X_test_rf, features_rf, best_n_rf, cv_rf = rf_importance_selection_cv(
        X_train, X_val, X_test, y_train, max_features=300, n_estimators_list=[100, 200]
    )
    results['RF_Importance'] = {
        'features': features_rf,
        'X_train': X_train_rf,
        'X_val': X_val_rf,
        'X_test': X_test_rf,
        'best_param': f'n_estimators={best_n_rf}'
    }
    evaluation_results['RF_Importance'] = {'cv_score': cv_rf[0], 'cv_std': cv_rf[1], 'n_features': len(features_rf)}
    
    # 方法3: 互信息
    X_train_mi, X_val_mi, X_test_mi, features_mi, best_k_mi, cv_mi = mutual_info_selection_cv(
        X_train, X_val, X_test, y_train, k_values=[200, 250, 300]
    )
    results['Mutual_Info'] = {
        'features': features_mi,
        'X_train': X_train_mi,
        'X_val': X_val_mi,
        'X_test': X_test_mi,
        'best_param': f'k={best_k_mi}'
    }
    evaluation_results['Mutual_Info'] = {'cv_score': cv_mi[0], 'cv_std': cv_mi[1], 'n_features': len(features_mi)}
    
    # 方法4: SelectKBest (F-test)
    X_train_kb, X_val_kb, X_test_kb, features_kb, best_k_kb, cv_kb = selectkbest_selection_cv(
        X_train, X_val, X_test, y_train, k_values=[200, 250, 300]
    )
    results['SelectKBest_Ftest'] = {
        'features': features_kb,
        'X_train': X_train_kb,
        'X_val': X_val_kb,
        'X_test': X_test_kb,
        'best_param': f'k={best_k_kb}'
    }
    evaluation_results['SelectKBest_Ftest'] = {'cv_score': cv_kb[0], 'cv_std': cv_kb[1], 'n_features': len(features_kb)}
    
    # 方法5: SelectPercentile
    X_train_pct, X_val_pct, X_test_pct, features_pct, best_pct, cv_pct = select_percentile_selection(
        X_train, X_val, X_test, y_train, percentile_values=[5, 10, 15, 20]
    )
    results['SelectPercentile'] = {
        'features': features_pct,
        'X_train': X_train_pct,
        'X_val': X_val_pct,
        'X_test': X_test_pct,
        'best_param': f'percentile={best_pct}'
    }
    evaluation_results['SelectPercentile'] = {'cv_score': cv_pct[0], 'cv_std': cv_pct[1], 'n_features': len(features_pct)}
    
    # 打印特征选择方法对比
    print("\n" + "=" * 70)
    print("特征选择方法对比")
    print("=" * 70)
    print(f"{'方法':<25} {'CV ROC-AUC':<20} {'特征数':<10}")
    print("-" * 55)
    
    # 按CV分数排序
    sorted_methods = sorted(evaluation_results.items(), key=lambda x: x[1]['cv_score'], reverse=True)
    for method, eval_result in sorted_methods:
        print(f"{method:<25} {eval_result['cv_score']:.4f} ± {eval_result['cv_std']:<12} {eval_result['n_features']}")
    
    # 自动选择最佳方法
    best_method_name = sorted_methods[0][0]
    best_cv_score = sorted_methods[0][1]['cv_score']
    
    print("\n" + "=" * 70)
    print("自动选择最佳特征选择方法")
    print("=" * 70)
    print(f"\n✓ 最佳方法: {best_method_name}")
    print(f"✓ CV ROC-AUC: {best_cv_score:.4f}")
    
    final_features = results[best_method_name]['features']
    final_X_train = results[best_method_name]['X_train']
    final_X_val = results[best_method_name]['X_val']
    final_X_test = results[best_method_name]['X_test']
    
    # 保存选择的特征
    feature_df = pd.DataFrame({'feature': final_features})
    feature_df.to_csv(f"{OUTPUT_DIR}/selected_features.csv", index=False)
    print(f"✓ 选择的特征已保存至: {OUTPUT_DIR}/selected_features.csv")
    
    # 保存特征选择元数据
    metadata = {
        'best_method': best_method_name,
        'best_cv_score': float(best_cv_score),
        'n_features_selected': len(final_features),
        'all_methods_evaluation': {
            k: {
                'cv_score': float(v['cv_score']),
                'cv_std': float(v['cv_std']),
                'n_features': v['n_features'],
                'best_param': results[k].get('best_param', 'N/A')
            }
            for k, v in evaluation_results.items()
        },
        'random_state': RANDOM_STATE,
        'execution_time_seconds': round(time.time() - start_time, 2)
    }
    
    with open(f"{OUTPUT_DIR}/feature_selection_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ 特征选择元数据已保存至: {OUTPUT_DIR}/feature_selection_metadata.json")
    
    # 保存处理后的数据
    if isinstance(final_X_train, pd.DataFrame):
        train_sel = pd.concat([final_X_train.reset_index(drop=True), 
                               y_train.reset_index(drop=True)], axis=1)
        val_sel = pd.concat([final_X_val.reset_index(drop=True), 
                             y_val.reset_index(drop=True)], axis=1)
        test_sel = pd.concat([final_X_test.reset_index(drop=True), 
                              y_test.reset_index(drop=True)], axis=1)
    else:
        train_sel = pd.DataFrame(np.array(final_X_train), columns=final_features)
        train_sel['label'] = np.array(y_train)
        val_sel = pd.DataFrame(np.array(final_X_val), columns=final_features)
        val_sel['label'] = np.array(y_val)
        test_sel = pd.DataFrame(np.array(final_X_test), columns=final_features)
        test_sel['label'] = np.array(y_test)
    
    train_sel.to_csv(f"{OUTPUT_DIR}/train_data_selected.csv", index=False)
    val_sel.to_csv(f"{OUTPUT_DIR}/val_data_selected.csv", index=False)
    test_sel.to_csv(f"{OUTPUT_DIR}/test_data_selected.csv", index=False)
    
    print(f"✓ 处理后训练数据: {OUTPUT_DIR}/train_data_selected.csv")
    print(f"✓ 处理后验证数据: {OUTPUT_DIR}/val_data_selected.csv")
    print(f"✓ 处理后测试数据: {OUTPUT_DIR}/test_data_selected.csv")
    
    elapsed_time = time.time() - start_time
    print(f"\n总执行时间: {elapsed_time:.2f}秒")
    
    return final_features, final_X_train, final_X_val, final_X_test, y_train, y_val, y_test, results, evaluation_results


if __name__ == "__main__":
    final_features, X_train, X_val, X_test, y_train, y_val, y_test, results, evaluation_results = main()
    print("\n" + "=" * 70)
    print("特征选择完成!")
    print("=" * 70)
    print(f"最终使用特征数: {len(final_features)}")