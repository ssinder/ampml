"""Command line interface for AMP-ML.

Three subcommands:

``ampml predict``
    Score peptide sequences (FASTA or a feature CSV) with the trained
    Gradient Boosting model.

``ampml benchmark``
    Print the test-set metrics that are recorded in ``results/``.

``ampml verify``
    Run the five-layer verification audit.

All subcommands work from any working directory: paths are resolved relative to
the repository root, which is inferred from this package's location.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = RESULTS_DIR / "models"

DEFAULT_MODEL = MODEL_DIR / "gradient_boosting_cv_model.pkl"
DEFAULT_FEATURES = DATA_DIR / "selected_features.csv"

__all__ = ["main"]


# ---------------------------------------------------------------- utilities


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file into ``(header, sequence)`` pairs."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)

    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def _resolve_repo_path(candidate: str) -> Path:
    """Resolve a user supplied path against cwd first, then the repo root."""
    as_path = Path(candidate).expanduser()
    if as_path.exists():
        return as_path.resolve()
    from_repo = (REPO_ROOT / as_path).resolve()
    if from_repo.exists():
        return from_repo
    raise FileNotFoundError(f"path not found: {candidate}")


# ------------------------------------------------------------------ predict


def _cmd_predict(args: argparse.Namespace) -> int:
    import pandas as pd

    model_path = _resolve_repo_path(args.model) if args.model else DEFAULT_MODEL
    if not model_path.exists():
        _eprint(f"error: model file not found: {model_path}")
        _eprint("hint: run `bash run_all.sh` to train the models, or pass --model.")
        return 2

    features_path = _resolve_repo_path(args.features) if args.features else DEFAULT_FEATURES
    if not features_path.exists():
        _eprint(f"error: selected feature list not found: {features_path}")
        _eprint("hint: run `python src/02_feature_selection.py` first.")
        return 2

    selected = pd.read_csv(features_path)["feature"].tolist()
    try:
        input_path = _resolve_repo_path(args.input)
    except FileNotFoundError as exc:
        _eprint(f"error: {exc}")
        return 2

    if input_path.suffix.lower() in {".fasta", ".fa", ".fsa", ".txt"}:
        records = _read_fasta(input_path)
        if not records:
            _eprint(f"error: no sequences found in {input_path}")
            return 2
        _eprint(f"read {len(records)} sequences from {input_path}")
        _eprint(
            "note: FASTA input requires the upstream feature extractor to produce "
            "descriptor rows. Use --input with a feature CSV for direct scoring, "
            "or run the pipeline in src/ to regenerate descriptors."
        )
        return 2

    frame = pd.read_csv(input_path)
    label_column = frame.columns[-1]
    if args.no_label or frame[label_column].nunique() > 10:
        features_frame = frame
        truth = None
    else:
        features_frame = frame.drop(columns=[label_column])
        truth = frame[label_column]

    missing = [f for f in selected if f not in features_frame.columns]
    if missing:
        _eprint(f"error: {len(missing)} required features are absent from the input")
        _eprint(f"first missing: {missing[:5]}")
        return 2

    matrix = features_frame[selected]
    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    predictions = model.predict(matrix)
    probabilities = model.predict_proba(matrix)

    out = pd.DataFrame(
        {
            "predicted_label": predictions,
            "probability_class_0": probabilities[:, 0],
            "probability_class_1": probabilities[:, 1],
        }
    )
    if truth is not None:
        out.insert(0, "true_label", truth.to_numpy())

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    positives = int((predictions == 1).sum())
    print(f"scored {len(predictions)} samples with {model_path.name}")
    print(f"predicted antimicrobial (L1): {positives} / {len(predictions)}")
    print(f"results written to {output_path}")

    if truth is not None:
        from sklearn.metrics import accuracy_score, f1_score

        accuracy = accuracy_score(truth.to_numpy(), predictions)
        f1 = f1_score(truth.to_numpy(), predictions)
        print(f"accuracy={accuracy:.4f}  f1={f1:.4f}")

    return 0


# ---------------------------------------------------------------- benchmark


def _cmd_benchmark(args: argparse.Namespace) -> int:
    comparison = RESULTS_DIR / "model_comparison_data.csv"
    if not comparison.exists():
        _eprint(f"error: {comparison} not found")
        return 2

    import pandas as pd

    frame = pd.read_csv(comparison)
    if args.format == "json":
        # Keep stdout machine-readable: the audit summary goes to stderr.
        print(frame.to_json(orient="records", indent=2))
    else:
        columns = ["Model", "Accuracy", "F1-Score", "ROC-AUC"]
        columns = [c for c in columns if c in frame.columns]
        print(frame[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    report = RESULTS_DIR / "verification_report.json"
    if report.exists():
        payload = json.loads(report.read_text(encoding="utf-8"))
        _eprint(f"\nverification audit: {report.relative_to(REPO_ROOT)}")
        _eprint(f"  generated: {payload.get('audit_time', 'n/a')}")
    return 0


# ------------------------------------------------------------------- verify


def _cmd_verify(args: argparse.Namespace) -> int:
    script = SRC_DIR / "99_verify_test_metrics.py"
    if not script.exists():
        _eprint(f"error: {script} not found")
        return 2
    completed = subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, check=False)
    return completed.returncode


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ampml",
        description="AMP-ML: reproducible antimicrobial peptide classification benchmark.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    subparsers = parser.add_subparsers(dest="command")

    predict = subparsers.add_parser("predict", help="score sequences with the trained model")
    predict.add_argument("--input", required=True, help="feature CSV to score (or a FASTA file)")
    predict.add_argument("--model", help="path to a pickled model (default: Gradient Boosting)")
    predict.add_argument("--features", help="path to the selected-feature list")
    predict.add_argument("--output", default="predictions.csv", help="output CSV path")
    predict.add_argument(
        "--no-label",
        action="store_true",
        help="treat every column of the input as a feature (no trailing label column)",
    )
    predict.set_defaults(func=_cmd_predict)

    benchmark = subparsers.add_parser("benchmark", help="print recorded test-set metrics")
    benchmark.add_argument("--format", choices=["table", "json"], default="table")
    benchmark.set_defaults(func=_cmd_benchmark)

    verify = subparsers.add_parser("verify", help="run the five-layer verification audit")
    verify.set_defaults(func=_cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(__version__)
        return 0

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
