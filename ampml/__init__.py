"""AMP-ML: reproducible machine learning benchmark for antimicrobial peptide classification.

This package provides a thin, dependency-light command line interface on top of
the research pipeline in ``src/``. The pipeline itself is intentionally kept as
plain numbered scripts so that every step is auditable; this package only wraps
them for interactive use.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
