"""05 -- simple moving average trend-filter momentum, vs bt + vectorbt.

simple moving average trend filter. on each monthly rebalance, each asset
is checked against its 200-day SMA. assets trading above their SMA receive
equal weight; assets below get zero weight (cash). if all assets are below
their SMAs, the portfolio goes 100% to cash. lookback: 200 trading days.
rebalance: monthly. costs: 15 bps + 3 bps. this tests signal-driven
allocation where the number of active positions varies over time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import RESULTS_ROOT, T212_COMMISSION, T212_SLIPPAGE, month_starts, run_multi_bucket
from utils.data import BACKTEST_START
from utils.engine import run_benchmark
from utils.strategies import sma_momentum


def run_single(close: pd.DataFrame, spx: pd.Series, results_dir: Path) -> None:
    """run sma-momentum for one bucket (lookback uses pre-2020 data, backtests 2020+)."""
    rebal = month_starts(close)
    ws = sma_momentum(close, rebal, lookback=200)
    close_bt = close.loc[close.index >= BACKTEST_START]
    spx_bt = spx.loc[spx.index >= BACKTEST_START]
    ws_bt = {d: w for d, w in ws.items() if d >= BACKTEST_START}
    run_benchmark(
        ws_bt, close_bt,
        results_dir=results_dir,
        title="05 SMA momentum (200-day)",
        commission=T212_COMMISSION,
        slippage=T212_SLIPPAGE,
        note="SMA trend filter; lookback on 2018-2020, backtest 2020-2025",
        spx=spx_bt,
    )


def main():
    run_multi_bucket(run_single, RESULTS_ROOT / "05-sma-momentum", n_assets=6,
                     full_history=True)


if __name__ == "__main__":
    main()
