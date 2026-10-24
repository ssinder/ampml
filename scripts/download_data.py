#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download the upstream feature table that anchors the whole pipeline.

The 78 MB ``all_features_combined_standard.csv`` is distributed as a GitHub
Release asset rather than being committed to git, so that ``git clone`` stays
fast. This script fetches it and verifies its MD5 against the checksum the
project has been using since the first audited run.

Usage:

    python scripts/download_data.py              # download and verify
    python scripts/download_data.py --force      # re-download even if present
    python scripts/download_data.py --check      # verify only, never download
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "rawdata" / "all_features_combined_standard.csv"

EXPECTED_MD5 = "7ce771dc55f1cf66d0dd28f8609c0bcb"

# Primary: GitHub Release asset. Mirrors below are fallbacks for networks that
# cannot reach github.com directly.
URLS = [
    "https://github.com/ssinder/ampml/releases/download/v0.1.0/all_features_combined_standard.csv",
    "https://github.com/ssinder/ampml/releases/latest/download/all_features_combined_standard.csv",
]


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path) -> bool:
    if not path.exists():
        print(f"not present: {path}")
        return False
    actual = md5_of(path)
    if actual == EXPECTED_MD5:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"OK  {path.name}  ({size_mb:.1f} MB, md5 {actual})")
        return True
    print(f"MD5 MISMATCH for {path.name}")
    print(f"  expected {EXPECTED_MD5}")
    print(f"  actual   {actual}")
    return False


def download(url: str, destination: Path) -> bool:
    print(f"downloading from {url}")
    print("  (78 MB — this takes a moment on a slow connection)")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = response.headers.get("Content-Length")
            total_mb = int(total) / (1024 * 1024) if total else 0
            downloaded = 0
            with destination.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total_mb:
                        pct = downloaded / (1024 * 1024) / total_mb * 100
                        print(f"\r  {downloaded / (1024 * 1024):.1f} / {total_mb:.1f} MB ({pct:.0f}%)", end="")
    except (urllib.error.URLError, OSError) as exc:
        print(f"\nfailed: {exc}")
        return False
    print()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    parser.add_argument("--check", action="store_true", help="verify the checksum without downloading")
    args = parser.parse_args()

    if args.check:
        return 0 if verify(TARGET) else 1

    if TARGET.exists() and not args.force:
        print(f"{TARGET.name} already present; verifying")
        if verify(TARGET):
            return 0
        print("re-downloading because the checksum did not match")
        args.force = True

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    tmp = TARGET.with_suffix(".csv.part")
    for url in URLS:
        if download(url, tmp):
            tmp.replace(TARGET)
            if verify(TARGET):
                print("\nDone. Next: python src/00_check_environment.py")
                return 0
            print("checksum mismatch after download; the file may be corrupted")
            return 1

    print("\nAll download sources failed.")
    print("Alternative: the file is also available from the project owner, or it can")
    print("be regenerated from the upstream snapshot in ampmodel/feature_extraction/.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
