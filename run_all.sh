#!/usr/bin/env bash
#
# AMP-ML 一键复现流水线
# ============================================================================
# 用途：供第三方验收人员用单条命令从原始数据复现全部结果
#
# 用法：
#   bash run_all.sh              完整流程（含环境自检）
#   bash run_all.sh --skip-check 跳过环境自检
#   bash run_all.sh --from 02    从指定阶段开始（01-99）
#   bash run_all.sh --only 99    只跑指定阶段
#   bash run_all.sh --list       列出所有阶段
#   bash run_all.sh --clean      清除 data/ 与 results/ 后重跑
#
# 退出码：0=全部成功；非 0=某阶段失败（脚本会立即停止）
# ============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---- Python 解释器定位 ----
if [ -x "$ROOT/venv/bin/python" ]; then
    PY="$ROOT/venv/bin/python"
elif [ -x "$ROOT/.venv/bin/python" ]; then
    PY="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    echo "✗ 未找到 Python 解释器。请先创建虚拟环境：" >&2
    echo "    python3 -m venv venv && ./venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_all_$(date +%Y%m%d_%H%M%S).log"

# ---- 阶段定义 ----
declare -a STAGES=(
    "check:00_check_environment.py:环境自检"
    "01:01_data_preparation.py:数据预处理与划分"
    "02:02_feature_selection.py:特征选择 4921→300"
    "03:03_model_training_cv.py:六模型训练与5折交叉验证"
    "03b:03b_generate_cv_curves.py:交叉验证曲线数据"
    "04b:04b_generate_figures_optimized.py:学术图表生成"
    "05:05_shap_analysis.py:SHAP可解释性分析"
    "99:99_verify_test_metrics.py:验收指标复算审计"
)

SKIP_CHECK=0
FROM_STAGE=""
ONLY_STAGE=""
DO_CLEAN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-check) SKIP_CHECK=1; shift ;;
        --from) FROM_STAGE="$2"; shift 2 ;;
        --only) ONLY_STAGE="$2"; shift 2 ;;
        --clean)  DO_CLEAN=1; shift ;;
        --list)
            echo "可用阶段："
            for s in "${STAGES[@]}"; do
                IFS=':' read -r id script desc <<< "$s"
                printf "  %-6s %-34s %s\n" "$id" "$script" "$desc"
            done
            exit 0 ;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "未知参数: $1（用 --help 查看用法）" >&2; exit 2 ;;
    esac
done

# ---- 清理 ----
if [ "$DO_CLEAN" = "1" ]; then
    echo "▸ 清理旧产物..."
    rm -rf "$ROOT/data" "$ROOT/results" "$ROOT/figures"
    mkdir -p "$ROOT/data" "$ROOT/results" "$ROOT/figures"
fi

started=0
[ -n "$FROM_STAGE" ] && started=1
[ -n "$ONLY_STAGE" ] && started=1

banner() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "  $2"
    echo "════════════════════════════════════════════════════════════"
}

echo "AMP-ML 一键复现流水线"
echo "项目根目录: $ROOT"
echo "Python    : $PY ($("$PY" -V 2>&1))"
echo "日志文件  : $RUN_LOG"

for s in "${STAGES[@]}"; do
    IFS=':' read -r id script desc <<< "$s"

    # --only 过滤
    if [ -n "$ONLY_STAGE" ] && [ "$id" != "$ONLY_STAGE" ]; then continue; fi

    # --from 之前的阶段跳过（check 始终执行，除非 --skip-check）
    if [ "$started" = "1" ] && [ -n "$FROM_STAGE" ] && [ "$id" != "check" ]; then
        if [ "$id" = "$FROM_STAGE" ]; then started=0; else continue; fi
    fi

    # check 阶段可跳过
    if [ "$id" = "check" ] && [ "$SKIP_CHECK" = "1" ]; then
        echo "▸ 跳过环境自检（--skip-check）"
        continue
    fi

    if [ ! -f "$ROOT/src/$script" ]; then
        echo "✗ 脚本不存在: src/$script" >&2
        exit 1
    fi

    banner "$id — $desc" "src/$script"
    t0=$(date +%s)

    if "$PY" "$ROOT/src/$script" 2>&1 | tee -a "$RUN_LOG"; then
        t1=$(date +%s)
        printf "✓ %s 完成（%ds）\n" "$id" "$((t1 - t0))"
    else
        echo ""
        echo "✗ $id ($script) 执行失败，已停止。" >&2
        echo "  完整日志: $RUN_LOG" >&2
        exit 1
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  流水线执行完毕"
echo "════════════════════════════════════════════════════════════"
echo "完整日志: $RUN_LOG"
echo ""
echo "关键产物："
echo "  results/cv_results_summary.csv       交叉验证指标汇总"
echo "  results/model_comparison_data.csv    测试集模型对比"
echo "  results/verification_report.json     验收审计报告"
echo ""
echo "查看验收结论："
echo "  cat results/verification_report.json"
echo "════════════════════════════════════════════════════════════"
