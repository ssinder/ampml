# Contributing to AMP-ML

感谢你考虑参与贡献。本文档说明如何复现、验证与提交改动。

> 中文说明见本文档下方 [中文版](#中文贡献指南)。English-first, 中文在后。

---

## Quick reference

| I want to... | Command |
|---|---|
| Check my environment | `python src/00_check_environment.py` |
| Reproduce everything from scratch | `bash run_all.sh --clean` |
| Run only the verification audit | `bash run_all.sh --only 99` |
| Run the test suite | `pytest -q` |
| Lint my changes | `pre-commit run --all-files` |
| See which stages exist | `bash run_all.sh --list` |

---

## Environment setup

```bash
python3.13 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python src/00_check_environment.py   # must pass before anything else
```

`00_check_environment.py` is the single gate for the whole pipeline. If it fails,
no downstream script will produce trustworthy output. Fix the environment first.

---

## Repository layout

```
ampml/
├── src/            # downstream ML pipeline (00 → 99), numbered by execution order
├── ampmodel/       # upstream: data construction + feature extraction
├── data/           # generated intermediate data (git-ignored, reproducible)
├── rawdata/        # upstream feature table (the reproduction entry point)
├── results/        # metrics, models, SHAP outputs, verification reports
├── figures/        # publication-ready figures
├── docs/           # technical report, reproducibility report, verification manual
└── test/           # pytest suite
```

---

## The five engineering contracts

These are **not** style preferences. Breaking them silently invalidates published numbers.

### 1. Single source of random seed

`src/config.py` → `RANDOM_STATE = 42`. **Never** define a seed inline in a business
script. Import it:

```python
from config import RANDOM_STATE
```

Changing the seed changes every downstream metric. It is a high-risk change and
must be reflected in `results/training_metadata.json`.

### 2. Column-order contract

The label column must be **the last column** of every data CSV. Downstream code
universally uses `iloc[:, :-1]` for features and `iloc[:, -1]` for the label.

### 3. Path contract

All paths derive from `config.py`'s `ROOT`. **No hardcoded absolute paths.**
Scripts must run correctly from any working directory:

```bash
python /any/path/to/ampml/src/03_model_training_cv.py
```

### 4. Sample IDs stay out of the main CSV

Adding an ID column would break the column-order contract. Sample IDs live in
`data/sample_id_mapping.csv` instead.

### 5. `results/` and `data/` CSVs are the single source of truth

Numbers in documentation are **generated** from these files. Never hand-copy a
metric into a README, a report, or a figure caption.

---

## Contributing a change

1. **Open an issue first** for anything beyond a trivial fix — especially for
   changes that touch the data split, the feature selection, or model hyperparameters.

2. **Branch** from `main`:
   ```bash
   git checkout -b feature/your-change
   ```

3. **Make sure it reproduces.** If your change affects the pipeline, run:
   ```bash
   bash run_all.sh --from 03 --only 03,99
   ```
   The verification audit (`99`) must still pass all 30 checks.

4. **Add tests** for new logic in `test/`.

5. **Run the quality gates**:
   ```bash
   pre-commit run --all-files
   pytest -q
   ```

6. **Open a PR** using the template. The template asks whether your change alters
   reported metrics — if it does, explain why, and include the before/after numbers.

---

## Reporting a bug

Please use the issue templates. Include:

- Your OS, Python version, and `pip list` output for the relevant packages
- The exact command you ran
- The full traceback
- The output of `python src/00_check_environment.py`

---

## Code of Conduct

Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

---

# 中文贡献指南

## 环境准备

```bash
python3.13 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python src/00_check_environment.py   # 必须先通过
```

## 五条工程铁律

1. **随机种子唯一来源** = `src/config.py` 的 `RANDOM_STATE`（=42）。业务脚本**不得**自行定义，
   必须 `from config import RANDOM_STATE` 导入。改种子会改变全部下游指标，属高风险变更。
2. **列序契约**：数据 CSV 的标签列必须是**最后一列**。下游统一 `iloc[:, :-1]` 取特征、`iloc[:, -1]` 取标签。
3. **路径契约**：路径一律由 `config.py` 的 `ROOT` 推导，**禁止硬编码绝对路径**。
   脚本必须能从任意工作目录执行。
4. **样本 ID 不入主 CSV**（会破坏列序契约），另存 `data/sample_id_mapping.csv`。
5. **`results/` 与 `data/` 下的 CSV/JSON 是数值唯一权威来源**，文档中的表格由其生成，
   **禁止手工誊抄**数值。

## 提交改动前必须验证

```bash
bash run_all.sh --from 03 --only 03,99   # 五层审计 30 项必须全部通过
pre-commit run --all-files
pytest -q
```

## 提交 PR

请使用 PR 模板。模板中会问你：**这次改动是否影响已报告的指标数值**？
若影响，需说明原因并附上改动前后的数值对比。

## 报告 bug

请使用 issue 模板，并附上：操作系统、Python 版本、相关包的 `pip list` 输出、
你执行的完整命令、完整报错堆栈，以及 `python src/00_check_environment.py` 的输出。

---

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
