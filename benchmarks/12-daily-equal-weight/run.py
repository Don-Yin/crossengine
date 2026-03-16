"""12 -- daily equal-weight rebalance, vs bt + vectorbt.

daily equal-weight rebalance (frequency amplification test). identical logic
to benchmark 01 (equal weight 1/N across all assets) but rebalanced every
trading day instead of monthly. this ~20x increase in rebalance frequency
tests whether engine divergence accumulates proportionally with trading
frequency, isolating the effect of execution-count on implementation risk.
rebalance: daily. costs: 15 bps + 3 bps.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import RESULTS_ROOT, T212_COMMISSION, T212_SLIPPAGE, every_day, run_multi_bucket
from utils.engine import run_benchmark
from utils.strategies import daily_equal_weight


def run_single(close: pd.DataFrame, spx: pd.Series, results_dir: Path) -> None:
    """run daily equal-weight for one bucket's worth of assets."""
    rebal = every_day(close)
    ws = daily_equal_weight(close, rebal)
    run_benchmark(
        ws, close,
        results_dir=results_dir,
        title="12 daily equal-weight rebalance",
        commission=T212_COMMISSION,
        slippage=T212_SLIPPAGE,
        note="daily rebalancing; frequency amplification test; all engines receive identical total cost rate",
        spx=spx,
    )


def main():
    run_multi_bucket(run_single, RESULTS_ROOT / "12-daily-equal-weight", n_assets=6)


if __name__ == "__main__":
    main()
