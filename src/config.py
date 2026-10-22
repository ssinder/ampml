"""
AMP-ML 全局配置

所有脚本共享的路径、随机种子与超参上限。
修改此文件即影响全链路，改动请同步更新 results/training_metadata.json。

复现要点：
  1. RANDOM_STATE 固定为 42，任何脚本不得自行定义随机种子
  2. 路径一律基于项目根目录推导，禁止硬编码绝对路径
  3. RAW_SHA256 用于校验上游特征文件是否被篡改
"""

from pathlib import Path

# ===== 路径 =====
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = RESULTS_DIR / "models"
FIGURES_DIR = ROOT / "figures"
RAWDATA_DIR = ROOT / "rawdata"

# 上游 ampmodel 产出的特征文件
RAW_FEATURES = RAWDATA_DIR / "all_features_combined_standard.csv"

# 原始数据校验和（2026-10-19 实测）
# 上游 ampmodel/feature_extraction/features/results/ 下同名文件 MD5 一致
RAW_FEATURES_MD5 = "7ce771dc55f1cf66d0dd28f8609c0bcb"

# ===== 全局随机种子 =====
# 单一事实来源：src/ 下所有脚本必须从此处导入
RANDOM_STATE = 42

# ===== 数据划分 =====
# 划分参数改动会直接改变全部下游指标，属高风险变更
SPLIT_SEED = RANDOM_STATE
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# ===== 交叉验证 =====
CV_FOLDS = 5
CV_INNER_FOLDS = 3
CV_SEED = RANDOM_STATE
CV_OUTER_SEED = RANDOM_STATE + 1  # 嵌套 CV 外层，避免内外层切分重合

# ===== 特征选择 =====
TARGET_N_FEATURES = 300
FS_SEED = RANDOM_STATE

# ===== 元数据输出一致性 =====
# 下游脚本统一以最后一列作为标签，列名必须保持为 "label"
LABEL_COLUMN = "label"
SAMPLE_ID_COLUMN = "ID"


def ensure_dir(path: Path) -> Path:
    """确保目录存在并返回该路径"""
    path.mkdir(parents=True, exist_ok=True)
    return path
