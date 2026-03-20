"""shared test fixtures with lazy data download.

e2e tests require price data (data/close.parquet) and bucket definitions
(data/buckets.json) which are gitignored. this module downloads them
automatically on first run via yfinance + the curation pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"

sys.path.insert(0, str(PROJECT / "benchmarks"))
sys.path.insert(0, str(PROJECT / "src"))


def _data_available() -> bool:
    """check if all required data files exist."""
    return (DATA_DIR / "close.parquet").exists() and (DATA_DIR / "buckets.json").exists()


def _download_data() -> None:
    """download price data and generate buckets if missing."""
    import subprocess

    python = sys.executable

    # step 1: download prices (requires universe.json)
    if not (DATA_DIR / "close.parquet").exists():
        if not (DATA_DIR / "universe.json").exists():
            pytest.skip("data/universe.json not found; run data/curation/build_universe.py first")
        print("\n  downloading price data via yfinance (first run only)...")
        result = subprocess.run(
            [python, str(DATA_DIR / "curation" / "download_yfinance.py")],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            pytest.skip(f"data download failed: {result.stderr[:200]}")

    # step 2: generate buckets if missing
    if not (DATA_DIR / "buckets.json").exists():
        if not (DATA_DIR / "close.parquet").exists():
            pytest.skip("data/close.parquet not available after download")
        print("\n  generating stratified buckets (first run only)...")
        result = subprocess.run(
            [python, str(DATA_DIR / "curation" / "stratification.py")],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            pytest.skip(f"bucket generation failed: {result.stderr[:200]}")


@pytest.fixture(scope="session")
def test_data():
    """fixture that ensures price data and buckets are available.

    usage in e2e tests:
        def test_something(test_data):
            close = test_data["close"]
            buckets = test_data["buckets"]
    """
    if not _data_available():
        _download_data()

    if not _data_available():
        pytest.skip("required data files not available")

    import pandas as pd

    close = pd.read_parquet(DATA_DIR / "close.parquet")
    buckets = json.loads((DATA_DIR / "buckets.json").read_text())

    return {"close": close, "buckets": buckets, "data_dir": DATA_DIR}
