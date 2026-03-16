"""09 -- daily binary switch across stratified buckets.

on each trading day, puts 100% of the portfolio into one of two assets,
alternating daily (day 1: 100% asset A, day 2: 100% asset B, day 3:
100% asset A, ...).  zero transaction costs -- this is a pure
execution-order ablation that isolates how engines handle extreme daily
turnover and concentrated positions.

runs across all stratified buckets, taking the first 2 tickers from
each bucket.

params:
  - allocation = 100% in one asset, 0% in the other, alternating daily
  - rebalance frequency = every trading day
  - cost model = zero (0 bps commission + 0 bps slippage)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import RESULTS_ROOT, every_day, run_multi_bucket
from utils.engine import run_benchmark
from utils.strategies import daily_binary_switch


# -- main ---------------------------------------------------------------------

def run_single(close: pd.DataFrame, spx: pd.Series, results_dir: Path) -> None:
    """run daily binary switch for one bucket pair."""
    asset_a, asset_b = close.columns[0], close.columns[1]
    rebal = every_day(close)
    ws = daily_binary_switch(close, rebal, asset_a=asset_a, asset_b=asset_b)
    run_benchmark(
        ws, close,
        results_dir=results_dir,
        title=f"09 daily binary switch ({asset_a}/{asset_b})",
        commission=0.0,
        slippage=0.0,
        note=f"pair {asset_a}/{asset_b}; zero costs; execution-order ablation",
        spx=spx,
    )


def main():
    """run across all buckets."""
    run_multi_bucket(run_single, RESULTS_ROOT / "09-daily-binary-switch", n_assets=2)


if __name__ == "__main__":
    main()
