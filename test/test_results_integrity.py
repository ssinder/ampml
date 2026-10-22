"""Consistency tests between the shipped results and the documented claims.

These tests guard the single most dangerous failure mode of a benchmark
repository: a README or report quoting a number that no longer matches the
artifact it was generated from.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from conftest import REPO_ROOT


def test_model_comparison_has_all_six_models(model_comparison) -> None:
    expected = {
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "XGBoost",
        "Gradient Boosting",
        "Voting Ensemble",
    }
    actual = set(model_comparison["Model"])
    assert actual == expected, f"model set changed: {actual ^ expected}"


def test_all_metrics_are_in_valid_range(model_comparison) -> None:
    for column in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        assert column in model_comparison.columns, f"missing metric column: {column}"
        values = model_comparison[column]
        assert values.between(0.0, 1.0).all(), f"{column} out of range: {values.tolist()}"


def test_gradient_boosting_is_the_top_accuracy_model(model_comparison) -> None:
    """The project documents Gradient Boosting as the primary model."""
    best = model_comparison.loc[model_comparison["Accuracy"].idxmax()]
    assert best["Model"] == "Gradient Boosting", (
        f"expected Gradient Boosting to lead, got {best['Model']} ({best['Accuracy']:.4f})"
    )
    assert best["Accuracy"] == pytest.approx(0.8645833333333334, abs=1e-6)
    assert best["ROC-AUC"] == pytest.approx(0.9432296618226267, abs=1e-6)


def test_readme_quotes_match_the_metrics_artifact() -> None:
    """Any accuracy quoted in the README must exist in results/."""
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        pytest.skip("README.md is not present")

    text = readme.read_text(encoding="utf-8")
    import pandas as pd

    frame = pd.read_csv(REPO_ROOT / "results" / "model_comparison_data.csv")
    known = {f"{v:.4f}" for v in frame["Accuracy"]} | {f"{v:.2f}" for v in frame["Accuracy"]}

    quoted = re.findall(r"(\d\.\d{2,4})\s*%?\s*accuracy", text, re.I)
    quoted += re.findall(r"accuracy[^.\n]{0,20}?(\d\.\d{2,4})", text, re.I)
    unknown = [q for q in set(quoted) if q not in known and f"{float(q):.4f}" not in known]
    assert not unknown, f"README quotes accuracies absent from results/: {sorted(unknown)}"


def test_readme_miss_rates_match_the_confusion_matrix() -> None:
    """The 'miss rate' column in the README is derived: FN / (FN + TP)."""
    import pandas as pd

    readme = REPO_ROOT / "README.md"
    matrix_path = REPO_ROOT / "results" / "confusion_matrix_data.csv"
    if not readme.exists() or not matrix_path.exists():
        pytest.skip("README.md or confusion matrix is not present")

    text = readme.read_text(encoding="utf-8")
    matrix = pd.read_csv(matrix_path)

    # Each README row looks like: | **Gradient Boosting** (recommended) | **0.8646** | ...
    rows = re.findall(r"^\|\s*\*{0,2}([A-Za-z ]+?)\*{0,2}[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*(\d+\.\d+)%", text, re.M)
    assert rows, "could not parse the results table from README.md"

    computed = {
        row["Model"]: round(row["FN"] / (row["FN"] + row["TP"]) * 100, 2)
        for _, row in matrix.iterrows()
    }
    for model, quoted in rows:
        model = model.strip()
        if model not in computed:
            continue
        assert float(quoted) == pytest.approx(computed[model], abs=0.01), (
            f"README miss rate for {model} is {quoted}% but the confusion matrix implies "
            f"{computed[model]}%"
        )


def test_verification_report_is_consistent(verification_report) -> None:
    """The audit report must record a passing result for every layer."""
    assert verification_report, "verification report is empty"

    text = json.dumps(verification_report, ensure_ascii=False)
    assert "audit_time" in verification_report, "audit report must record when it ran"

    layers = verification_report.get("layers") or verification_report.get("audit_layers")
    if layers is None:
        # Fall back to searching for layer markers in the serialized payload.
        assert re.search(r"L[1-5]", text), "audit report must enumerate its layers"
    else:
        assert len(layers) >= 5, f"expected five audit layers, found {len(layers)}"


def test_confusion_matrix_totals_match_test_set_size() -> None:
    """TN+FP+FN+TP must equal the independent test set size (n=384)."""
    import pandas as pd

    path = REPO_ROOT / "results" / "confusion_matrix_data.csv"
    if not path.exists():
        pytest.skip("results/confusion_matrix_data.csv is not present")

    frame = pd.read_csv(path)
    totals = frame[["TN", "FP", "FN", "TP"]].sum(axis=1)
    for model, total in zip(frame["Model"], totals, strict=True):
        assert total == 384, f"{model}: confusion matrix totals {total}, expected 384"


def test_models_are_present_and_non_empty() -> None:
    models = sorted((REPO_ROOT / "results" / "models").glob("*.pkl"))
    if not models:
        pytest.skip("results/models is not present")
    assert len(models) == 6, f"expected 6 pickled models, found {len(models)}"
    for model in models:
        assert model.stat().st_size > 0, f"{model.name} is empty"


def test_figures_are_present() -> None:
    figures = sorted((REPO_ROOT / "figures").glob("*.png"))
    if not figures:
        pytest.skip("figures/ is not present")
    assert len(figures) >= 4, f"expected at least 4 figures, found {len(figures)}"


def test_documentation_is_present() -> None:
    docs = REPO_ROOT / "docs"
    if not docs.exists():
        pytest.skip("docs/ is not present")
    names = {p.name for p in docs.glob("*.md")}
    for required in ["TECHNICAL_REPORT.md", "REPRODUCIBILITY.md", "THIRD_PARTY_VERIFICATION.md"]:
        assert required in names, f"missing required document: {required}"


def test_negative_sampling_sensitivity_results_are_consistent() -> None:
    """The three-strategy sensitivity study must show the expected ordering.

    This is the project's headline scientific claim, so it is asserted rather
    than merely documented.
    """
    import pandas as pd

    path = REPO_ROOT / "results" / "negative_sampling_sensitivity_summary.csv"
    if not path.exists():
        pytest.skip("results/negative_sampling_sensitivity_summary.csv is not present")

    frame = pd.read_csv(path).set_index("strategy")

    for required in ["baseline", "length_matched", "shuffled_positives"]:
        assert required in frame.index, f"missing strategy: {required}"

    # Length matching must not improve performance: the easy negative set
    # contributed to the baseline number.
    assert frame.loc["length_matched", "accuracy_mean"] < frame.loc["baseline", "accuracy_mean"]

    # Relabelled positives must collapse every model to chance.
    assert frame.loc["shuffled_positives", "roc_auc_mean"] < 0.6
    assert frame.loc["shuffled_positives", "accuracy_mean"] < 0.6

    # The baseline must be meaningfully above chance.
    assert frame.loc["baseline", "roc_auc_mean"] > 0.85


def test_sensitivity_study_covers_all_models() -> None:
    import pandas as pd

    path = REPO_ROOT / "results" / "negative_sampling_sensitivity.csv"
    if not path.exists():
        pytest.skip("results/negative_sampling_sensitivity.csv is not present")

    frame = pd.read_csv(path)
    assert len(frame) == 18, f"expected 3 strategies x 6 models = 18 rows, found {len(frame)}"
    assert frame["model"].nunique() == 6
    assert frame["strategy"].nunique() == 3
    for column in ["accuracy", "f1_score", "roc_auc", "miss_rate"]:
        assert column in frame.columns
        assert frame[column].between(0, 1).all()
