#!/usr/bin/env python3
"""
99_verify_test_metrics.py —— 第三方验收审计脚本（只读核对，不重训）

用途
----
在独立测试集上加载 results/models/ 下已交付的模型，重新计算全部分类指标，
与 results/model_comparison_data.csv 中记录的数值逐项核对，并输出一份
可独立阅读的审计报告，供第三方验收人员使用。

本脚本不执行任何训练或超参搜索，仅做推理与指标复算。

五层递进审计
------------
  L1 手算交叉验证   用混淆矩阵手算指标，与 sklearn 结果对拍，排除库调用错误
  L2 数据溯源       校验测试集规模、类别分布、特征数、样本 ID 映射完整性
  L3 模型文件指纹   记录各 pkl 的 MD5，证明审计对象与交付对象为同一文件
  L4 指标比对       复算指标 vs 交付记录值，逐项核对（容差 1e-6）
  L5 达标判定       按验收阈值（准确率 ≥ 0.80）给出最终结论

用法
----
  python src/99_verify_test_metrics.py
  python src/99_verify_test_metrics.py --json          # 额外输出 JSON 到 stdout
  python src/99_verify_test_metrics.py --threshold 0.85

输出
----
  results/verification_report.json    机器可读审计报告
  results/VERIFICATION_SUMMARY.md     人可读审计摘要
"""

import argparse
import hashlib
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

warnings.filterwarnings("ignore")

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from config import (  # noqa: E402
    ROOT, DATA_DIR, RESULTS_DIR, MODEL_DIR, RANDOM_STATE,
)

TEST_PATH = DATA_DIR / "test_data_selected.csv"
MAPPING_PATH = DATA_DIR / "sample_id_mapping.csv"
COMPARE_PATH = RESULTS_DIR / "model_comparison_data.csv"
TRAIN_META = RESULTS_DIR / "training_metadata.json"
REPORT_PATH = RESULTS_DIR / "verification_report.json"
SUMMARY_PATH = RESULTS_DIR / "VERIFICATION_SUMMARY.md"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression_cv_model.pkl",
    "Random Forest": "random_forest_cv_model.pkl",
    "SVM": "svm_cv_model.pkl",
    "XGBoost": "xgboost_cv_model.pkl",
    "Gradient Boosting": "gradient_boosting_cv_model.pkl",
    "Voting Ensemble": "voting_ensemble_cv_model.pkl",
}

TOLERANCE = 1e-6
DEFAULT_THRESHOLD = 0.80
PRIMARY_MODEL = "Gradient Boosting"


def md5_of(path: Path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ===================== L1 手算交叉验证 =====================
def manual_metrics(tn, fp, fn, tp):
    """用混淆矩阵四格数直接手算各项指标，不依赖 sklearn"""
    total = tn + fp + fn + tp
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
    }


# ===================== L2 数据溯源 =====================
def audit_data(report):
    print("\n" + "=" * 70)
    print("L2  数据溯源校验")
    print("=" * 70)

    info = {}

    df = pd.read_csv(TEST_PATH)
    n, p = df.shape[0], df.shape[1] - 1
    y = df.iloc[:, -1]
    dist = {int(k): int(v) for k, v in y.value_counts().items()}
    print(f"  测试集规模      : {n} 样本 × {p} 特征")
    print(f"  类别分布        : L0={dist.get(0, 0)}  L1={dist.get(1, 0)}")
    print(f"  标签列名        : {df.columns[-1]}")

    checks = []

    ok = (n == 384)
    checks.append({"item": "测试集样本数", "expected": 384, "actual": n, "pass": ok})
    print(f"  [{'OK' if ok else 'NG'}] 样本数 = 384（实测 {n}）")

    ok = (p == 300)
    checks.append({"item": "特征维度", "expected": 300, "actual": p, "pass": ok})
    print(f"  [{'OK' if ok else 'NG'}] 特征数 = 300（实测 {p}）")

    ok = (dist.get(0, 0) == 185 and dist.get(1, 0) == 199)
    checks.append({"item": "类别分布", "expected": "L0=185,L1=199",
                   "actual": f"L0={dist.get(0,0)},L1={dist.get(1,0)}", "pass": ok})
    print(f"  [{'OK' if ok else 'NG'}] 类别分布 = 185/199（实测 {dist.get(0,0)}/{dist.get(1,0)}）")

    # 样本 ID 映射
    if MAPPING_PATH.exists():
        m = pd.read_csv(MAPPING_PATH)
        test_rows = m[m["split"] == "test"]
        id_ok = len(m) == 2557 and m["ID"].nunique() == 2557 and len(test_rows) == n
        checks.append({"item": "样本ID映射", "expected": "2557唯一ID, test段384行",
                       "actual": f"{len(m)}行/{m['ID'].nunique()}唯一/test{len(test_rows)}行",
                       "pass": bool(id_ok)})
        print(f"  [{'OK' if id_ok else 'NG'}] 样本ID映射：{len(m)} 行 / "
              f"{m['ID'].nunique()} 唯一 ID / test 段 {len(test_rows)} 行")

        # ID 前缀与标签一致性
        pref_ok = (((m["ID"].str[:2] == "L0") & (m["label"] == 0)) |
                   ((m["ID"].str[:2] == "L1") & (m["label"] == 1))).all()
        checks.append({"item": "ID前缀与标签一致", "expected": "100%一致",
                       "actual": "100%一致" if pref_ok else "存在不一致",
                       "pass": bool(pref_ok)})
        print(f"  [{'OK' if pref_ok else 'NG'}] ID 前缀与标签一致性")
    else:
        checks.append({"item": "样本ID映射", "expected": "存在", "actual": "缺失",
                       "pass": False})
        print(f"  [NG] 样本ID映射文件缺失：{MAPPING_PATH.name}")

    info.update({
        "test_set_size": n, "n_features": p, "class_distribution": dist,
        "checks": checks, "all_pass": all(c["pass"] for c in checks),
    })
    return info


# ===================== L3 模型文件指纹 =====================
def audit_models(report):
    print("\n" + "=" * 70)
    print("L3  模型文件指纹")
    print("=" * 70)

    fps = {}
    for name, fname in MODEL_FILES.items():
        p = MODEL_DIR / fname
        if p.exists():
            h = md5_of(p)
            fps[name] = {"file": fname, "md5": h, "size_bytes": p.stat().st_size}
            print(f"  {name:22s} {h}  {p.stat().st_size/1024:.0f} KB")
        else:
            fps[name] = {"file": fname, "md5": None, "size_bytes": 0}
            print(f"  {name:22s} 文件缺失")

    print("\n  说明：以上 MD5 即本次审计所验证的模型文件指纹。")
    print("  第三方可自行计算 pkl 的 MD5 与此比对，确认审计对象与交付对象同一。")
    return fps


# ===================== L4+L5 指标比对与达标判定 =====================
def audit_metrics(recorded, report, fps, threshold):
    print("\n" + "=" * 70)
    print("L4/L5  指标复算与达标判定")
    print("=" * 70)

    all_match = True
    l1_ok = True
    results = {}

    for model_name, fname in MODEL_FILES.items():
        print(f"\n  ── {model_name}")

        with open(MODEL_DIR / fname, "rb") as f:
            model = pickle.load(f)

        X_test = report["_X"]
        y_test = report["_y"]

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        computed = {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "F1-Score": float(f1_score(y_test, y_pred, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, y_proba)),
        }

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in cm.ravel())

        # L1：手算对拍
        manual = manual_metrics(tn, fp, fn, tp)
        l1_diffs = {k: abs(manual[k] - computed[k]) for k in manual}
        l1_pass = all(d < TOLERANCE for d in l1_diffs.values())
        l1_ok &= l1_pass
        worst = max(l1_diffs.values())
        print(f"    L1手算对拍    : {'一致' if l1_pass else '不一致'} (最大差 {worst:.2e})")

        # L4：与交付记录比对
        ref = recorded.loc[model_name]
        comparison = {}
        for key, val in computed.items():
            rec_val = float(ref[key])
            diff = abs(val - rec_val)
            matched = diff < TOLERANCE
            all_match &= matched
            comparison[key] = {
                "computed": round(val, 10),
                "recorded": round(rec_val, 10),
                "abs_diff": float(f"{diff:.3e}"),
                "match": bool(matched),
            }
            print(f"    {key:<12} 复算={val:.6f}  记录={rec_val:.6f}  "
                  f"[{'OK' if matched else 'MISMATCH'}]")

        # L5：达标判定
        meets = computed["Accuracy"] >= threshold
        print(f"    混淆矩阵      : TN={tn} FP={fp} FN={fn} TP={tp}")
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        print(f"    漏检率(FN/(FN+TP)) = {fnr*100:.2f}%")
        print(f"    L5准确率≥{threshold:.0%}   : {'达标' if meets else '未达标'}")

        results[model_name] = {
            "metrics": comparison,
            "manual_recompute": {k: round(v, 10) for k, v in manual.items()},
            "manual_vs_sklearn_max_diff": float(f"{worst:.2e}"),
            "manual_check_pass": bool(l1_pass),
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
            "miss_rate_positive": round(fnr, 6),
            "model_md5": fps[model_name]["md5"],
            "all_metrics_match": bool(all(v["match"] for v in comparison.values())),
            "meets_threshold": bool(meets),
        }

    return results, all_match, l1_ok


def write_summary(report, threshold):
    """生成人类可读的审计摘要"""
    L = []
    A = L.append
    s = report["summary"]

    A("# 验收审计摘要\n")
    A(f"生成时间：{report['audit_time']}\n")
    A(f"审计脚本：`src/99_verify_test_metrics.py`\n")
    A(f"验收阈值：准确率 ≥ {threshold:.0%}\n")
    A("---\n")

    A("## 一、审计结论\n")
    verdict = "✅ 通过" if s["all_pass"] else "❌ 未通过"
    A(f"**{verdict}** — {s['verdict']}\n")

    A("| 审计层 | 内容 | 结果 |")
    A("|--------|------|------|")
    A(f"| L1 | 混淆矩阵手算 vs sklearn 交叉验证 | {'✅ 一致' if s['l1_pass'] else '❌ 存在差异'} |")
    A(f"| L2 | 数据溯源（规模/分布/ID映射） | {'✅ 通过' if s['l2_pass'] else '❌ 未通过'} |")
    A(f"| L3 | 模型文件指纹记录 | ✅ 已记录（{len(s['model_count'])} 个） |")
    A(f"| L4 | 复算指标 vs 交付记录 | {'✅ 全部一致' if s['l4_pass'] else '❌ 存在不一致'} |")
    A(f"| L5 | 达标判定（≥{threshold:.0%}） | {'✅ 达标' if s['l5_pass'] else '❌ 未达标'} |")

    A(f"\n共核对 **{s['n_items_checked']}** 项数值，容差 {TOLERANCE:g}。\n")

    A("## 二、达标情况\n")
    A("| 模型 | 准确率 | F1 | ROC-AUC | 混淆矩阵 (TN/FP/FN/TP) | 漏检率 | 达标 |")
    A("|------|--------|-----|---------|---------------------------|--------|------|")
    for name, r in report["models"].items():
        m = r["metrics"]
        cm = r["confusion_matrix"]
        A(f"| {name} | {m['Accuracy']['computed']:.4f} | {m['F1-Score']['computed']:.4f} | "
          f"{m['ROC-AUC']['computed']:.4f} | {cm['TN']}/{cm['FP']}/{cm['FN']}/{cm['TP']} | "
          f"{r['miss_rate_positive']*100:.2f}% | {'✅' if r['meets_threshold'] else '❌'} |")

    best = max(report["models"].items(), key=lambda kv: kv[1]["metrics"]["Accuracy"]["computed"])
    A(f"\n最高准确率：**{best[0]} {best[1]['metrics']['Accuracy']['computed']:.4f}**\n")
    A(f"主推交付模型：**{PRIMARY_MODEL}**"
      f"（准确率 {report['models'][PRIMARY_MODEL]['metrics']['Accuracy']['computed']:.4f}，"
      f"ROC-AUC {report['models'][PRIMARY_MODEL]['metrics']['ROC-AUC']['computed']:.4f}）\n")

    A("> 漏检率 = FN / (FN + TP)，即阳性样本中被误判为阴性的比例。\n")

    A("## 三、数据溯源\n")
    d = report["data_audit"]
    A(f"- 测试集：{d['test_set_size']} 样本 × {d['n_features']} 特征")
    A(f"- 类别分布：L0={d['class_distribution'].get(0,0)} / L1={d['class_distribution'].get(1,0)}")
    A(f"- 全局随机种子：{RANDOM_STATE}")
    for c in d["checks"]:
        A(f"- [{'✓' if c['pass'] else '✗'}] {c['item']}：期望 {c['expected']}，实测 {c['actual']}")

    A("\n## 四、模型文件指纹\n")
    A("第三方可自行计算以下 MD5，确认审计对象与交付对象为同一文件。\n")
    A("| 模型 | 文件 | MD5 |")
    A("|------|------|-----|")
    for name, f in report["model_files"].items():
        A(f"| {name} | `{f['file']}` | `{f['md5']}` |")

    A("\n## 五、审计边界声明\n")
    A("本审计**仅验证**：已交付模型在已交付测试集上的指标可复算一致。\n")
    A("本审计**不验证**：")
    A("- 训练过程是否使用了测试集信息（超参搜索仅在训练集内进行，代码可查）")
    A("- 上游原始数据库到本数据集的构建过程（存在已记录的不可复现环节）")
    A("- 模型对全新肽段的泛化能力（需独立外部数据集）")
    A("\n详见 `docs/REPRODUCIBILITY.md` 与 `docs/THIRD_PARTY_VERIFICATION.md`。\n")

    SUMMARY_PATH.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="AMP-ML 第三方验收审计")
    ap.add_argument("--json", action="store_true", help="额外将报告 JSON 打到 stdout")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"准确率验收阈值（默认 {DEFAULT_THRESHOLD:.0%}）")
    args = ap.parse_args()

    print("=" * 70)
    print("AMP-ML 第三方验收审计")
    print(f"验收阈值：准确率 ≥ {args.threshold:.0%}")
    print("=" * 70)

    import datetime
    report = {
        "audit_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "verification_purpose": "第三方验收审计 — 测试集指标复算，不重训",
        "threshold": args.threshold,
        "tolerance": TOLERANCE,
        "random_state": RANDOM_STATE,
    }

    # 加载测试集
    test_df = pd.read_csv(TEST_PATH)
    report["_X"] = test_df.iloc[:, :-1]
    report["_y"] = test_df.iloc[:, -1]

    # L2
    data_audit = audit_data(report)

    # L3
    model_files = audit_models(report)

    # L4/L5
    recorded = pd.read_csv(COMPARE_PATH).set_index("Model")
    model_results, l4_pass, l1_pass = audit_metrics(recorded, report, model_files, args.threshold)

    # 清理临时键
    report.pop("_X", None)
    report.pop("_y", None)

    n_items = sum(len(r["metrics"]) for r in model_results.values())
    n_match = sum(sum(1 for v in r["metrics"].values() if v["match"])
                  for r in model_results.values())
    l5_pass = all(r["meets_threshold"] for r in model_results.values())

    all_pass = all([l1_pass, data_audit["all_pass"], l4_pass, l5_pass])

    report["data_audit"] = data_audit
    report["model_files"] = model_files
    report["models"] = model_results
    report["summary"] = {
        "l1_pass": bool(l1_pass),
        "l2_pass": bool(data_audit["all_pass"]),
        "l4_pass": bool(l4_pass),
        "l5_pass": bool(l5_pass),
        "all_pass": bool(all_pass),
        "n_items_checked": n_items,
        "n_items_matched": n_match,
        "model_count": list(model_files.keys()),
        "verdict": (
            "五层审计全部通过：手算一致、数据溯源完整、指标完全对得上、全部模型达标"
            if all_pass else
            "存在未通过项，详见报告各层 check 明细"
        ),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    write_summary(report, args.threshold)

    print("\n" + "=" * 70)
    print("审计结论汇总")
    print("=" * 70)
    print(f"  L1 手算交叉验证   : {'通过' if l1_pass else '未通过'}")
    print(f"  L2 数据溯源       : {'通过' if data_audit['all_pass'] else '未通过'}")
    print(f"  L3 模型指纹       : 已记录 {len(model_files)} 个")
    print(f"  L4 指标比对       : {n_match}/{n_items} 项一致")
    print(f"  L5 达标判定       : {'全部达标' if l5_pass else '存在未达标模型'}")
    print("-" * 70)
    print(f"  最终结论: {report['summary']['verdict']}")
    print("-" * 70)
    print(f"  报告(JSON) : {REPORT_PATH}")
    print(f"  摘要(MD)   : {SUMMARY_PATH}")
    print("=" * 70)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
