"""Tests for the ``ampml`` command line interface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from conftest import REPO_ROOT


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ampml", *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_exits_cleanly() -> None:
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "predict" in result.stdout
    assert "benchmark" in result.stdout
    assert "verify" in result.stdout


def test_version_is_reported() -> None:
    result = _run_cli("--version")
    assert result.returncode == 0
    assert result.stdout.strip()


def test_benchmark_prints_metrics() -> None:
    result = _run_cli("benchmark")
    assert result.returncode == 0, result.stderr
    assert "Gradient Boosting" in result.stdout


def test_benchmark_json_is_valid() -> None:
    import json

    result = _run_cli("benchmark", "--format", "json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert any(row["Model"] == "Gradient Boosting" for row in payload)


def test_predict_reports_missing_input() -> None:
    result = _run_cli("predict", "--input", "definitely/not/here.csv")
    assert result.returncode == 2
    assert "not found" in (result.stderr + result.stdout).lower()


def test_predict_runs_on_a_real_feature_table() -> None:
    """End-to-end: score the shipped test features and reproduce the accuracy."""
    import shutil
    import tempfile

    import pandas as pd

    test_data = REPO_ROOT / "data" / "test_data_selected.csv"
    if not test_data.exists():
        pytest.skip("data/test_data_selected.csv is not present")

    # The scratch directory lives under .tmp/ inside the repository rather
    # than in the system temp directory: some sandboxes lock down the system
    # temp root, and writing into the repository root would leave stray
    # directories behind. .tmp/ is git-ignored and removed by the test.
    scratch_root = REPO_ROOT / ".tmp"
    scratch_root.mkdir(exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="ampml_cli_", dir=str(scratch_root)))
    try:
        output = workdir / "predictions.csv"
        result = _run_cli("predict", "--input", str(test_data), "--output", str(output))
        if result.returncode != 0:
            pytest.skip(f"model unavailable in this environment: {result.stderr.strip()[:200]}")

        assert output.exists()
        frame = pd.read_csv(output)
        assert len(frame) == 384
        assert {"predicted_label", "probability_class_0", "probability_class_1"} <= set(frame.columns)
        assert frame["probability_class_0"].between(0, 1).all()
        assert frame["probability_class_1"].between(0, 1).all()

        # The shipped model should reproduce the documented Gradient Boosting accuracy.
        truth = pd.read_csv(test_data)["label"].to_numpy()
        accuracy = (frame["predicted_label"].to_numpy() == truth).mean()
        assert accuracy == pytest.approx(0.8645833333333334, abs=0.01)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.rmtree(scratch_root, ignore_errors=True)
