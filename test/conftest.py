"""Shared pytest fixtures for the AMP-ML test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"
SRC_DIR = REPO_ROOT / "src"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def results_dir() -> Path:
    return RESULTS_DIR


@pytest.fixture(scope="session")
def model_comparison():
    """The recorded independent-test-set metrics (the acceptance reference)."""
    import pandas as pd

    path = RESULTS_DIR / "model_comparison_data.csv"
    if not path.exists():
        pytest.skip("results/model_comparison_data.csv is not present")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def verification_report() -> dict:
    path = RESULTS_DIR / "verification_report.json"
    if not path.exists():
        pytest.skip("results/verification_report.json is not present")
    return json.loads(path.read_text(encoding="utf-8"))
