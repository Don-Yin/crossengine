"""07 -- cross-sectional momentum (12-1 month), vs bt + vectorbt.

cross-sectional momentum (12-1). on each monthly rebalance, assets are
ranked by their trailing 252-day return (approximately 12 months), skipping
the most recent 21 days (1 month) to avoid short-term reversal. the top 2
assets by past return receive equal weight (50% each); all others receive
zero. formation period: 252 trading days. skip period: 21 days. top-k: 2.
rebalance: monthly. costs: 15 bps + 3 bps. this tests a classic momentum
signal where asset selection changes over time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import RESULTS_ROOT, T212_COMMISSION, T212_SLIPPAGE, month_starts, run_multi_bucket
from utils.data import BACKTEST_START
from utils.engine import run_benchmark
from utils.strategies import cross_sectional_momentum


def run_single(close: pd.DataFrame, spx: pd.Series, results_dir: Path) -> None:
    """run cross-sectional momentum for one bucket (lookback uses pre-2020 data, backtests 2020+)."""
    rebal = month_starts(close)
    ws = cross_sectional_momentum(close, rebal, formation=252, skip=21, top_k=2)
    close_bt = close.loc[close.index >= BACKTEST_START]
    spx_bt = spx.loc[spx.index >= BACKTEST_START]
    ws_bt = {d: w for d, w in ws.items() if d >= BACKTEST_START}
    run_benchmark(ws_bt, close_bt, results_dir=results_dir,
                  title="07 cross-sectional momentum (12-1, top 2)",
                  commission=T212_COMMISSION, slippage=T212_SLIPPAGE,
                  note="12-1 month momentum; lookback on 2018-2020, backtest 2020-2025",
                  spx=spx_bt)


def main():
    """run across all buckets."""
    run_multi_bucket(run_single, RESULTS_ROOT / "07-cross-momentum", n_assets=6,
                     full_history=True)


if __name__ == "__main__":
    main()
