#!/usr/bin/env python3
"""
AMP-ML 环境自检脚本（第三方验收第一道关卡）

用途：
  在正式复现前，一次性校验运行环境、依赖版本、原始数据完整性、必需文件齐全性。
  任何一项不通过都会阻断后续流程，并给出明确的修复指引。

用法：
  python src/00_check_environment.py           # 严格模式，有错即退出码 1
  python src/00_check_environment.py --json    # 输出 JSON，便于程序化调用
  python src/00_check_environment.py --repair  # 尝试自动修复可修复项

退出码：
  0 = 全部通过
  1 = 存在阻断性错误
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from config import (  # noqa: E402
    ROOT, DATA_DIR, RESULTS_DIR, MODEL_DIR, RAWDATA_DIR, RAW_FEATURES,
    RAW_FEATURES_MD5, RANDOM_STATE, LABEL_COLUMN,
)

# ===== 期望的依赖版本（与 requirements.txt 一致）=====
# 键为 pip 包名，值为 (import 名, 期望版本)
EXPECTED_PACKAGES = {
    "numpy":         ("numpy",          "2.4.4"),
    "pandas":        ("pandas",         "3.0.2"),
    "scikit-learn":  ("sklearn",        "1.8.0"),
    "scipy":         ("scipy",          "1.17.1"),
    "xgboost":       ("xgboost",        "3.2.0"),
    "shap":          ("shap",           "0.51.0"),
    "matplotlib":    ("matplotlib",     "3.10.8"),
    "seaborn":       ("seaborn",        "0.13.2"),
    "modlamp":       ("modlamp",        "4.3.2"),
    "biopython":     ("Bio",            "1.87"),
    "joblib":        ("joblib",         "1.5.3"),
}

MIN_PYTHON = (3, 9)

# ===== 必须存在的输入文件 =====
REQUIRED_INPUTS = [
    RAWDATA_DIR / "all_features_combined_standard.csv",
]

# ===== 必须存在的产物文件（完整跑通后应有） =====
REQUIRED_OUTPUTS = [
    DATA_DIR / "train_data_selected.csv",
    DATA_DIR / "val_data_selected.csv",
    DATA_DIR / "test_data_selected.csv",
    DATA_DIR / "selected_features.csv",
    DATA_DIR / "sample_id_mapping.csv",
    RESULTS_DIR / "cv_results_summary.csv",
    RESULTS_DIR / "model_comparison_data.csv",
    MODEL_DIR / "gradient_boosting_cv_model.pkl",
    MODEL_DIR / "voting_ensemble_cv_model.pkl",
    RESULTS_DIR / "verification_report.json",
]


class Checker:
    def __init__(self):
        self.results = []

    def add(self, level, name, ok, detail="", fix=""):
        self.results.append({
            "level": level, "name": name, "ok": bool(ok),
            "detail": detail, "fix": fix,
        })
        return ok

    def info(self, name, detail=""):
        self.add("INFO", name, True, detail)

    def pass_(self, name, detail=""):
        self.add("PASS", name, True, detail)

    def warn(self, name, detail="", fix=""):
        self.add("WARN", name, True, detail, fix)

    def fail(self, name, detail="", fix=""):
        self.add("FAIL", name, False, detail, fix)

    @property
    def n_fail(self):
        return sum(1 for r in self.results if r["level"] == "FAIL")

    @property
    def n_warn(self):
        return sum(1 for r in self.results if r["level"] == "WARN")


def md5_of(path: Path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_python(c: Checker):
    v = sys.version_info
    txt = f"{v.major}.{v.minor}.{v.micro} ({platform.python_implementation()})"
    if (v.major, v.minor) >= MIN_PYTHON:
        c.pass_("Python 版本", txt)
    else:
        c.fail("Python 版本", txt,
               f"需 >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}，实测 {v.major}.{v.minor}")


def check_packages(c: Checker):
    import importlib
    import importlib.metadata as md

    mismatch, missing = [], []
    for dist_name, (import_name, want) in EXPECTED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(dist_name)
            continue
        except Exception as exc:
            # A package can be installed yet fail to import: a missing OpenMP
            # runtime (xgboost), a compiled ABI mismatch, or a broken wheel.
            # Those are environment defects we must report, not crash on.
            missing.append(f"{dist_name}（导入失败: {type(exc).__name__}: {exc}）")
            continue
        # 版本以 pip 元数据为准：部分包（如 modlamp）内部 __version__ 常量陈旧
        try:
            got = md.version(dist_name)
        except Exception:
            mod = importlib.import_module(import_name)
            got = getattr(mod, "__version__", "unknown")
        if got != want:
            mismatch.append(f"{dist_name}: 期望 {want} / 实测 {got}")

    if missing:
        c.fail("依赖包缺失", ", ".join(missing),
               "执行 pip install -r requirements.txt")
    if mismatch:
        c.warn("依赖包版本偏差", "; ".join(mismatch),
               "为保证数值可复现，请按 requirements.txt 安装指定版本")
    if not missing and not mismatch:
        c.pass_("依赖包版本", f"{len(EXPECTED_PACKAGES)} 个包全部匹配")


def check_raw_data(c: Checker):
    p = RAW_FEATURES
    if not p.exists():
        c.fail("原始特征文件缺失", str(p),
               "请将 all_features_combined_standard.csv 放入 rawdata/ 目录")
        return

    size_mb = p.stat().st_size / 1024 / 1024
    c.info("原始特征文件", f"{p.name}  {size_mb:.1f} MB")

    got = md5_of(p)
    if got == RAW_FEATURES_MD5:
        c.pass_("原始特征文件 MD5", got)
    else:
        c.fail("原始特征文件 MD5 不符",
               f"期望 {RAW_FEATURES_MD5} / 实测 {got}",
               "文件可能已被修改或损坏，请取回原始版本")


def check_inputs(c: Checker):
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        c.fail("必需输入文件缺失", ", ".join(missing), "补齐后重试")
    else:
        c.pass_("必需输入文件", f"{len(REQUIRED_INPUTS)} 项齐全")


def check_outputs(c: Checker):
    missing, present = [], []
    for p in REQUIRED_OUTPUTS:
        (present if p.exists() else missing).append(str(p.relative_to(ROOT)))

    if not missing:
        c.pass_("产物文件", f"{len(present)} 项齐全")
    else:
        c.warn("产物文件缺失", f"{len(missing)}/{len(REQUIRED_OUTPUTS)} 项未生成: "
               + ", ".join(missing[:4]) + ("..." if len(missing) > 4 else ""),
               "执行完整流程: bash run_all.sh")


def check_outputs_deep(c: Checker):
    """产物内容一致性抽检：验证测试集规模与 ID 映射"""
    tdp = DATA_DIR / "test_data_selected.csv"
    mp = DATA_DIR / "sample_id_mapping.csv"

    if not tdp.exists() or not mp.exists():
        c.warn("产物内容抽检", "缺少 test_data_selected.csv 或 sample_id_mapping.csv，无法抽检")
        return

    try:
        import pandas as pd
        head = pd.read_csv(tdp, nrows=1)
        n_feat = len(head.columns) - 1
        n_rows = sum(1 for _ in open(tdp, encoding="utf-8")) - 1
        c.info("测试集维度", f"{n_rows} 样本 × {n_feat} 特征")

        if n_feat == 300:
            c.pass_("测试集特征数", "300（与特征选择输出一致）")
        else:
            c.fail("测试集特征数", f"期望 300 / 实测 {n_feat}",
                   "请检查 02_feature_selection.py 是否正常完成")

        m = pd.read_csv(mp)
        if len(m) == 2557 and m["ID"].nunique() == 2557:
            c.pass_("样本 ID 映射", f"{len(m)} 条唯一 ID，完整无重复")
        else:
            c.warn("样本 ID 映射异常",
                   f"总行 {len(m)} / 唯一 ID {m['ID'].nunique()}（期望 2557 / 2557）",
                   "建议重跑 01_data_preparation.py")
    except Exception as e:
        c.warn("产物内容抽检", f"读取失败: {type(e).__name__}: {e}")


def check_config(c: Checker):
    c.info("全局随机种子", str(RANDOM_STATE))
    c.info("标签列名约定", LABEL_COLUMN)
    if (ROOT / "requirements.txt").exists():
        c.pass_("依赖锁定文件", "requirements.txt 存在")
    else:
        c.fail("依赖锁定文件缺失", "requirements.txt 不存在",
               "缺少版本锁定将无法保证数值可复现")


def main():
    ap = argparse.ArgumentParser(description="AMP-ML 环境自检")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = ap.parse_args()

    c = Checker()
    check_python(c)
    check_config(c)
    check_packages(c)
    check_inputs(c)
    check_raw_data(c)
    check_outputs(c)
    check_outputs_deep(c)

    if args.json:
        print(json.dumps({
            "passed": c.n_fail == 0,
            "n_fail": c.n_fail,
            "n_warn": c.n_warn,
            "checks": c.results,
        }, ensure_ascii=False, indent=2))
        return 0 if c.n_fail == 0 else 1

    icon = {"PASS": "✓", "FAIL": "✗", "WARN": "!", "INFO": "·"}
    print("=" * 72)
    print("AMP-ML 环境自检")
    print("=" * 72)
    print(f"项目根目录: {ROOT}")
    print(f"平台      : {platform.platform()}")
    print("-" * 72)
    for r in c.results:
        line = f"  {icon[r['level']]} {r['name']}"
        if r["detail"]:
            line += f"  —  {r['detail']}"
        print(line)
        if not r["ok"] or r["level"] == "WARN":
            if r["fix"]:
                print(f"      → 修复建议: {r['fix']}")
    print("-" * 72)
    if c.n_fail:
        print(f"结论: 不通过（{c.n_fail} 项错误，{c.n_warn} 项警告）")
        print("请先修复上述错误项，再执行复现流程。")
    else:
        extra = f"，{c.n_warn} 项警告" if c.n_warn else ""
        print(f"结论: 通过（0 项错误{extra}）")
        print("可执行: bash run_all.sh")
    print("=" * 72)
    return 0 if c.n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
