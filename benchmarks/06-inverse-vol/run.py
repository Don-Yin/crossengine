"""06 -- inverse-volatility weighting, vs bt + vectorbt.

inverse-volatility weighting. on each monthly rebalance, each asset's
weight is proportional to the inverse of its recent realised volatility:
w_i = (1/sigma_i) / sum(1/sigma_j). sigma is the standard deviation of
daily returns over a trailing 60-day window. assets with lower volatility
receive higher weights. lookback: 60 trading days. rebalance: monthly.
costs: 15 bps + 3 bps. this tests a risk-parity-style strategy where
weights depend on a backward-looking statistical estimate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import RESULTS_ROOT, T212_COMMISSION, T212_SLIPPAGE, month_starts, run_multi_bucket
from utils.data import BACKTEST_START
from utils.engine import run_benchmark
from utils.strategies import inverse_volatility


def run_single(close: pd.DataFrame, spx: pd.Series, results_dir: Path) -> None:
    """run inverse-volatility for one bucket (lookback uses pre-2020 data, backtests 2020+)."""
    rebal = month_starts(close)
    ws = inverse_volatility(close, rebal, vol_window=60)
    close_bt = close.loc[close.index >= BACKTEST_START]
    spx_bt = spx.loc[spx.index >= BACKTEST_START]
    ws_bt = {d: w for d, w in ws.items() if d >= BACKTEST_START}
    run_benchmark(ws_bt, close_bt, results_dir=results_dir,
                  title="06 inverse-volatility weighting (60d)",
                  commission=T212_COMMISSION, slippage=T212_SLIPPAGE,
                  note="inverse-vol weights; lookback on 2018-2020, backtest 2020-2025",
                  spx=spx_bt)


def main():
    """run across all buckets."""
    run_multi_bucket(run_single, RESULTS_ROOT / "06-inverse-vol", n_assets=6,
                     full_history=True)


if __name__ == "__main__":
    main()
