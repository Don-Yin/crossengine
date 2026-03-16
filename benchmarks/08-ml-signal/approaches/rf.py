"""08/rf -- random forest walk-forward across all buckets.

trains a RandomForestRegressor (bagged deep trees) on a rolling window
of past data to predict each asset's 21-day forward return. different
bias-variance profile from the shallow boosted trees in GBR -- uses
bagging instead of boosting and deeper individual trees. a gold-standard
ML benchmark per Gu et al. (2020).

features and target: same as 08/gbr (see gbr.py docstring).

params:
  - model = RandomForestRegressor (100 trees, max_depth=5)
  - training window = 6 months of rolling data
  - lookahead gap = 21 trading days between train end and prediction date
  - top_k = 2 (equal-weight the 2 highest predicted-return assets)
  - rebalance frequency = monthly (first trading day of each month)
  - cost model = Trading212 (15 bps commission + 3 bps slippage)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from utils import BACKTEST_START, RESULTS_ROOT, T212_COMMISSION, T212_SLIPPAGE, month_starts, run_multi_bucket
from utils.engine import run_benchmark
from utils.strategies import ml_signal


def run_single(close: pd.DataFrame, spx: pd.Series, results_dir: Path) -> None:
    """run random forest walk-forward for one bucket (trains on full history, backtests 2020+)."""
    rebal = month_starts(close)
    ws = ml_signal(
        close, rebal, top_k=2,
        model_factory=lambda: RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42,
        ),
    )
    close_bt = close.loc[close.index >= BACKTEST_START]
    spx_bt = spx.loc[spx.index >= BACKTEST_START]
    ws_bt = {d: w for d, w in ws.items() if d >= BACKTEST_START}
    run_benchmark(ws_bt, close_bt, results_dir=results_dir,
                  title="08 ml-signal-rf",
                  commission=T212_COMMISSION, slippage=T212_SLIPPAGE,
                  note="sklearn RandomForest walk-forward; trains on 2018-2020, backtests 2020-2025",
                  spx=spx_bt)


def main():
    """run across all buckets with full 7-year history for training."""
    run_multi_bucket(run_single, RESULTS_ROOT / "08-ml-signal" / "rf",
                     n_assets=6, full_history=True)


if __name__ == "__main__":
    main()
