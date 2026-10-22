#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
负样本采样敏感性分析（Negative-sample sensitivity analysis）
=============================================================

**科学动机**

AMP 二分类任务中，负样本（L0）的采集方式与正样本（L1）截然不同：
L0 来自 UniProt 全蛋白 ≤60 aa 片段，L1 来自 DRAMP/dbAMP/APD/CampDB 等
天然抗菌肽库。两者在序列长度分布、氨基酸组成、以及「是否为真实收录的
天然活性肽」上都不可比。模型可能学到的是**采集来源的伪影**，而非抗菌活性。

本脚本量化这一偏差对报告指标的影响。

**实验设计**

三种 L0 负样本策略 × 6 个模型 = 18 组指标：

| 策略 | 说明 | 目的 |
|------|------|------|
| `baseline` | 原始 UniProt 采集的 L0（作为对照） | 基线 |
| `length_matched` | 用 L1 的长度分布重采样 L0 | 消除长度分布这一混杂因素 |
| `shuffled_positives` | 用打乱的 L1 序列充当"负样本" | 上界探针：若标签与序列内容无关，模型应退化到接近随机 |

关键控制：三种策略共用**同一个固定测试集**（原始 test split），
因此策略之间唯一的差异就是**训练集中负样本的构成方式**。

**输出**

- `results/negative_sampling_sensitivity.csv`  — 18 组指标
- `results/negative_sampling_sensitivity_summary.csv` — 按策略聚合
- `figures/fig5_negative_sampling_sensitivity.png` — 对比图

**执行**

    python src/07_negative_sampling_sensitivity.py
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 统一配置来源：路径与随机种子由 src/config.py 单一维护
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    DATA_DIR as DATA_DIR_PATH,
    RESULTS_DIR,
    FIGURES_DIR,
    RANDOM_STATE as GLOBAL_RANDOM_STATE,
)

RANDOM_STATE = GLOBAL_RANDOM_STATE
N_FEATURES = 300

TRAIN_PATH = DATA_DIR_PATH / "train_data_selected.csv"
VAL_PATH = DATA_DIR_PATH / "val_data_selected.csv"
TEST_PATH = DATA_DIR_PATH / "test_data_selected.csv"
OUTPUT_CSV = RESULTS_DIR / "negative_sampling_sensitivity.csv"
SUMMARY_CSV = RESULTS_DIR / "negative_sampling_sensitivity_summary.csv"
FIGURE_PATH = FIGURES_DIR / "fig5_negative_sampling_sensitivity.png"

STRATEGIES = ["baseline", "length_matched", "shuffled_positives"]


# ----------------------------------------------------------------- helpers


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_models():
    """Six models with the same defaults as src/03_model_training_cv.py."""
    import xgboost as xgb
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    lr = LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1, max_iter=3000)
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    svm = SVC(random_state=RANDOM_STATE, probability=True)
    xgb_model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0,
    )
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)

    return {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "SVM": svm,
        "XGBoost": xgb_model,
        "Gradient Boosting": gb,
    }, (lr, rf, svm, xgb_model, gb)


def _load_split():
    """Load the three splits, returning (train_X, train_y, val_X, val_y, test_X, test_y)."""
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)

    def split(frame):
        y = frame.iloc[:, -1].to_numpy()
        X = frame.iloc[:, :-1].to_numpy(dtype=float)
        return X, y

    train_X, train_y = split(train)
    val_X, val_y = split(val)
    test_X, test_y = split(test)
    return train_X, train_y, val_X, val_y, test_X, test_y


def _resample_negatives(X_neg: np.ndarray, strategy: str, rng: np.random.Generator):
    """Return a negative-sample matrix built according to ``strategy``."""
    n = X_neg.shape[0]
    if strategy == "baseline":
        return X_neg
    if strategy == "length_matched":
        # Resample rows of X_neg so that the *length* distribution matches the
        # positives. Length is the first column (modlamp "Length").
        return X_neg
    if strategy == "shuffled_positives":
        return X_neg
    raise ValueError(f"unknown strategy: {strategy}")


def _apply_length_matching(X_neg, X_pos, rng):
    """Resample negative rows so the length distribution matches the positives.

    Length is the first selected feature (modlamp ``Length``), which is the
    dominant confounder between the two collection protocols. We sample negative
    rows with replacement using the positives' length distribution as the
    sampling weight.
    """
    neg_len = X_neg[:, 0]
    pos_len = X_pos[:, 0]

    # Map each negative length to a discrete bucket so that weights are
    # comparable across the two float distributions.
    lo = min(neg_len.min(), pos_len.min())
    hi = max(neg_len.max(), pos_len.max())
    bins = np.linspace(lo, hi, 21)
    n_buckets = len(bins) + 1  # np.digitize returns 0..len(bins)

    neg_idx = np.clip(np.digitize(neg_len, bins), 0, n_buckets - 1)
    pos_idx = np.clip(np.digitize(pos_len, bins), 0, n_buckets - 1)

    neg_hist = np.bincount(neg_idx, minlength=n_buckets).astype(float)
    pos_hist = np.bincount(pos_idx, minlength=n_buckets).astype(float)

    # Target counts proportional to the positive length profile.
    target = pos_hist / pos_hist.sum() * len(neg_len)

    # Sample within each bucket according to the target proportions.
    picks = []
    for bucket in range(n_buckets):
        members = np.where(neg_idx == bucket)[0]
        if members.size == 0:
            continue
        wanted = int(round(target[bucket]))
        if wanted <= 0:
            continue
        chosen = rng.choice(members, size=wanted, replace=True)
        picks.append(chosen)

    if not picks:
        return X_neg
    return X_neg[np.concatenate(picks)]


def _evaluate(model, X_train, y_train, X_test, y_test) -> dict:
    """Fit on train, evaluate on the fixed test split."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "miss_rate": fn / (fn + tp) if (fn + tp) else float("nan"),
    }


# -------------------------------------------------------------------- main


def main() -> int:
    _banner("负样本采样敏感性分析")
    print(f"随机种子: {RANDOM_STATE}")

    for path in (TRAIN_PATH, VAL_PATH, TEST_PATH):
        if not path.exists():
            print(f"错误: 缺少数据文件 {path}")
            print("请先运行 src/01_data_preparation.py 与 src/02_feature_selection.py")
            return 1

    train_X, train_y, val_X, val_y, test_X, test_y = _load_split()
    print(f"train: {train_X.shape}, val: {val_X.shape}, test: {test_X.shape}")
    print(f"train 类别分布: L0={int((train_y == 0).sum())} L1={int((train_y == 1).sum())}")

    neg_mask = train_y == 0
    pos_mask = train_y == 1
    X_neg_all, y_neg = train_X[neg_mask], train_y[neg_mask]
    X_pos_all = train_X[pos_mask]

    models, base_estimators = build_models()
    rows = []

    for strategy in STRATEGIES:
        _banner(f"策略: {strategy}")
        rng = np.random.default_rng(RANDOM_STATE)

        if strategy == "baseline":
            X_neg = X_neg_all
        elif strategy == "length_matched":
            X_neg = _apply_length_matching(X_neg_all, X_pos_all, rng)
        elif strategy == "shuffled_positives":
            # Upper-bound probe: relabel a copy of the positives as negatives.
            # The *content* is antimicrobial, only the label is inverted, so a
            # model that has learned real antimicrobial activity must collapse.
            X_neg = X_pos_all.copy()
            rng.shuffle(X_neg)
        else:
            raise ValueError(strategy)

        print(f"  负样本数: {X_neg_all.shape[0]} -> {X_neg.shape[0]}")
        print(f"  负样本长度均值: {X_neg[:, 0].mean():.2f}  正样本: {X_pos_all[:, 0].mean():.2f}")

        X_strategy = np.vstack([X_neg, X_pos_all])
        y_strategy = np.concatenate([np.zeros(len(X_neg), dtype=int), np.ones(len(X_pos_all), dtype=int)])

        # Keep the class balance of the baseline split.
        if len(X_neg) != len(X_neg_all):
            keep = rng.permutation(len(X_strategy))[: len(X_neg_all) + len(X_pos_all)]
            X_strategy = X_strategy[keep]
            y_strategy = y_strategy[keep]

        for model_name, estimator in models.items():
            start = time.time()
            metrics = _evaluate(estimator, X_strategy, y_strategy, test_X, test_y)
            elapsed = time.time() - start
            row = {"strategy": strategy, "model": model_name, **metrics}
            rows.append(row)
            print(
                f"  {model_name:22s} acc={metrics['accuracy']:.4f} "
                f"f1={metrics['f1_score']:.4f} auc={metrics['roc_auc']:.4f} "
                f"({elapsed:.1f}s)"
            )

        # Voting ensemble over the five base models.
        from sklearn.ensemble import VotingClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        voting = VotingClassifier(
            estimators=[(name.lower().replace(" ", "_"), est) for name, est in models.items()],
            voting="soft",
        )
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", voting)])
        metrics = _evaluate(pipeline, X_strategy, y_strategy, test_X, test_y)
        rows.append({"strategy": strategy, "model": "Voting Ensemble", **metrics})
        print(
            f"  {'Voting Ensemble':22s} acc={metrics['accuracy']:.4f} "
            f"f1={metrics['f1_score']:.4f} auc={metrics['roc_auc']:.4f}"
        )

    # ------------------------------------------------------------- write out
    frame = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_CSV, index=False)
    print(f"\n结果已保存: {OUTPUT_CSV}")

    summary = (
        frame.groupby("strategy")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1_score", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            miss_rate_mean=("miss_rate", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"汇总已保存: {SUMMARY_CSV}")
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    _make_figure(frame, summary)

    meta = {
        "seed": RANDOM_STATE,
        "n_features": N_FEATURES,
        "strategies": STRATEGIES,
        "fixed_test_set": True,
        "note": (
            "All strategies are evaluated on the identical held-out test split; "
            "only the composition of the training negatives changes."
        ),
    }
    (RESULTS_DIR / "negative_sampling_sensitivity_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


def _make_figure(frame: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"baseline": "#1D9E75", "length_matched": "#378ADD", "shuffled_positives": "#D85A30"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    models = list(frame["model"].unique())

    for ax, metric, title in zip(
        axes,
        ["accuracy", "f1_score", "roc_auc"],
        ["Accuracy", "F1-score", "ROC-AUC"],
    ):
        width = 0.26
        positions = np.arange(len(models))
        for i, strategy in enumerate(STRATEGIES):
            subset = frame[frame["strategy"] == strategy].set_index("model").reindex(models)
            ax.bar(
                positions + (i - 1) * width,
                subset[metric].to_numpy(),
                width,
                label=strategy,
                color=colors[strategy],
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=8)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].legend(frameon=False, fontsize=9, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.28))
    fig.suptitle(
        "Sensitivity of test-set performance to negative-sample construction",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    print(f"图表已保存: {FIGURE_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
