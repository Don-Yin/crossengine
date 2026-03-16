"""10 -- cash-starved settlement stress test, vs bt + vectorbt.

cash-starved settlement stress test. alternates monthly between two 3-asset
allocations: [60%/30%/10%] and [10%/30%/60%] across the first 3 assets
in the configured universe (determined by bucket). the large swing from 60% to 10% forces
the engine to sell one position and buy another in the same bar, testing
whether sell proceeds are available to fund the buy within a single time
step. rebalance: monthly. costs: 15 bps + 3 bps. this specifically tests
settlement-timing differences between engines.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import RESULTS_ROOT, T212_COMMISSION, T212_SLIPPAGE, month_starts, run_multi_bucket
from utils.engine import run_benchmark
from utils.strategies import cash_starved_settlement


def run_single(close: pd.DataFrame, spx: pd.Series, results_dir: Path) -> None:
    """run cash-starved settlement for one bucket."""
    rebal = month_starts(close)
    ws = cash_starved_settlement(close, rebal, assets=list(close.columns))
    run_benchmark(ws, close, results_dir=results_dir, title="10 cash-starved settlement (60/30/10 rotation)",
                  commission=T212_COMMISSION, slippage=T212_SLIPPAGE,
                  note="tests settlement timing; all engines receive identical total cost rate", spx=spx)


def main():
    """run across all buckets."""
    run_multi_bucket(run_single, RESULTS_ROOT / "10-cash-starved", n_assets=3)


if __name__ == "__main__":
    main()
