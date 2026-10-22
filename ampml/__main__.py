"""Allow ``python -m ampml`` as an alias for the ``ampml`` console script."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
