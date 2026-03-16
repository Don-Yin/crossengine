"""08/gbr -- gradient boosting regressor walk-forward across all buckets.

machine learning signal using sklearn GradientBoostingRegressor. walk-forward
procedure: on each monthly rebalance, a model is trained on a rolling window
of the past 6 months of features (21/63/126-day cumulative returns and
20/60-day rolling volatility) to predict 21-day forward returns. a 21-day
gap between training end and prediction date prevents lookahead bias. the
top 2 assets by predicted return receive equal weight. model: GBR with 50
trees, max depth 3. rebalance: monthly. costs: 15 bps + 3 bps.

data setup: 7 years downloaded (2018-2025). the first 2 years (2018-2019)
provide a dedicated training buffer. backtesting evaluates on 2020-2025
(same 5-year window as all non-ML strategies) for comparability.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import BACKTEST_START, RESULTS_ROOT, T212_COMMISSION, T212_SLIPPAGE, month_starts, run_multi_bucket
from utils.engine import run_benchmark
from utils.strategies import ml_signal


def run_single(close: pd.DataFrame, spx: pd.Series, results_dir: Path) -> None:
    """run gbr walk-forward for one bucket (trains on full history, backtests 2020+)."""
    rebal = month_starts(close)
    ws = ml_signal(close, rebal, top_k=2)
    close_bt = close.loc[close.index >= BACKTEST_START]
    spx_bt = spx.loc[spx.index >= BACKTEST_START]
    ws_bt = {d: w for d, w in ws.items() if d >= BACKTEST_START}
    run_benchmark(ws_bt, close_bt, results_dir=results_dir,
                  title="08 ml-signal-gbr",
                  commission=T212_COMMISSION, slippage=T212_SLIPPAGE,
                  note="sklearn GBR walk-forward; trains on 2018-2020, backtests 2020-2025",
                  spx=spx_bt)


def main():
    """run across all buckets with full 7-year history for training."""
    run_multi_bucket(run_single, RESULTS_ROOT / "08-ml-signal" / "gbr",
                     n_assets=6, full_history=True)


if __name__ == "__main__":
    main()
