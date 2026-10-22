"""Contract tests for the five engineering rules that keep the numbers reproducible.

These are not "does the code run" tests. They assert the invariants that, if
broken, would silently invalidate every reported metric.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from conftest import REPO_ROOT, SRC_DIR


def _iter_business_scripts() -> list[Path]:
    """Scripts that participate in the ML pipeline (00/99 included)."""
    return sorted(p for p in SRC_DIR.glob("*.py") if p.name != "config.py")


def test_config_defines_the_single_random_seed() -> None:
    """Rule 1: the seed must live in exactly one place."""
    config = (SRC_DIR / "config.py").read_text(encoding="utf-8")
    assert re.search(r"^RANDOM_STATE\s*=\s*\d+", config, re.M), "config.py must define RANDOM_STATE"

    offenders = []
    for script in _iter_business_scripts():
        text = script.read_text(encoding="utf-8")
        # A literal definition of RANDOM_STATE is a violation. An alias such as
        # `RANDOM_STATE = GLOBAL_RANDOM_STATE` (imported from config) is allowed,
        # because the value still has a single origin.
        if re.search(r"^\s*RANDOM_STATE\s*=\s*\d+", text, re.M):
            offenders.append(script.name)
    assert not offenders, f"these scripts hardcode a RANDOM_STATE literal: {offenders}"


def test_every_script_that_uses_a_seed_imports_it_from_config() -> None:
    """Any script mentioning RANDOM_STATE must obtain it from config.py."""
    offenders = []
    for script in _iter_business_scripts():
        text = script.read_text(encoding="utf-8")
        if "RANDOM_STATE" not in text:
            continue
        if "from config import" not in text:
            offenders.append(script.name)
    assert not offenders, f"these scripts use RANDOM_STATE without importing config: {offenders}"


def test_business_scripts_import_paths_from_config() -> None:
    """Rule 3: no hardcoded absolute paths in the pipeline."""
    absolute_path_pattern = re.compile(r"""["'](/Users/|/home/|/tmp/|[A-Z]:\\\\)""")
    offenders = []
    for script in _iter_business_scripts():
        text = script.read_text(encoding="utf-8")
        if absolute_path_pattern.search(text):
            offenders.append(script.name)
    assert not offenders, f"these scripts contain absolute paths: {offenders}"


def test_scripts_import_from_config_module() -> None:
    """Rule 3 (continued): path/seed consumers must import from config."""
    consumers = ["01", "02", "03", "03b", "04b", "05", "06"]
    for prefix in consumers:
        matches = list(SRC_DIR.glob(f"{prefix}*.py"))
        assert matches, f"expected src/{prefix}*.py to exist"
        for script in matches:
            text = script.read_text(encoding="utf-8")
            assert "from config import" in text, f"{script.name} must import from config.py"


def test_run_all_script_exists_and_is_executable() -> None:
    script = REPO_ROOT / "run_all.sh"
    assert script.exists()
    assert script.stat().st_mode & 0o111, "run_all.sh must be executable"


def test_label_column_is_last_in_data_csvs() -> None:
    """Rule 2: the label column must be the final column of every data CSV."""
    import pandas as pd

    targets = [
        REPO_ROOT / "data" / "test_data_selected.csv",
        REPO_ROOT / "data" / "selected_features.csv",
    ]
    for path in targets:
        if not path.exists():
            pytest.skip(f"{path} is not present")
        if path.name == "selected_features.csv":
            continue
        frame = pd.read_csv(path, nrows=5)
        assert frame.columns[-1] == "label", f"{path.name} last column must be 'label'"


def test_selected_feature_count_matches_config() -> None:
    """The 300-feature target in config.py must match the shipped artifact."""
    import pandas as pd

    config = (SRC_DIR / "config.py").read_text(encoding="utf-8")
    match = re.search(r"TARGET_N_FEATURES\s*=\s*(\d+)", config)
    assert match, "config.py must define TARGET_N_FEATURES"

    path = REPO_ROOT / "data" / "selected_features.csv"
    if not path.exists():
        pytest.skip("data/selected_features.csv is not present")
    frame = pd.read_csv(path)
    assert len(frame) == int(match.group(1)), (
        f"selected_features.csv has {len(frame)} rows but config says {match.group(1)}"
    )


def test_sample_id_mapping_is_not_empty() -> None:
    """Rule 4: sample IDs live in a separate mapping file."""
    path = REPO_ROOT / "data" / "sample_id_mapping.csv"
    if not path.exists():
        pytest.skip("data/sample_id_mapping.csv is not present")
    assert path.stat().st_size > 0


def test_all_pipeline_scripts_are_valid_python() -> None:
    """A syntax error in any stage would break `run_all.sh` mid-run."""
    for script in _iter_business_scripts():
        source = script.read_text(encoding="utf-8")
        ast.parse(source, filename=str(script))
