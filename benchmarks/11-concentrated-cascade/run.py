"""11 -- concentrated cascade with heavy costs, vs bt + vectorbt.

concentrated cascade with heavy costs. alternates monthly between putting
95% in the first asset (1.25% each in the rest) and 95% in the last asset
(1.25% each in the rest). the extreme concentration (95% of portfolio in
one stock) combined with heavy costs (50 bps commission + 10 bps slippage)
amplifies any differences in how engines compute commissions on large
position changes. rebalance: monthly. commission: 50 bps. slippage: 10 bps.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import RESULTS_ROOT, month_starts, run_multi_bucket
from utils.engine import run_benchmark
from utils.strategies import concentrated_cascade

HEAVY_COMMISSION = 0.0050
HEAVY_SLIPPAGE = 0.0010


def run_single(close: pd.DataFrame, spx: pd.Series, results_dir: Path) -> None:
    """run concentrated cascade for one bucket."""
    rebal = month_starts(close)
    ws = concentrated_cascade(close, rebal)
    run_benchmark(ws, close, results_dir=results_dir, title="11 concentrated cascade (95/1.25, heavy costs)",
                  commission=HEAVY_COMMISSION, slippage=HEAVY_SLIPPAGE,
                  note="50 bps commission + 10 bps slippage; all engines receive identical total cost rate", spx=spx)


def main():
    """run across all buckets."""
    run_multi_bucket(run_single, RESULTS_ROOT / "11-concentrated-cascade", n_assets=6)


if __name__ == "__main__":
    main()
