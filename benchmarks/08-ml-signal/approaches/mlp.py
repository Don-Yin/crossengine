"""08/mlp -- multi-layer perceptron neural network walk-forward across all buckets.

trains an MLPRegressor (two hidden layers) on a rolling window of past
data to predict each asset's 21-day forward return. neural networks
were the best-performing model class in Gu et al. (2020) and the
exclusive model in Wang (2024, Financial Innovation).

features and target: same as 08/gbr (see gbr.py docstring).

params:
  - model = MLPRegressor (hidden layers = 32, 16 neurons; max_iter=1000)
  - early_stopping = true (validation-based early stop to avoid overfit)
  - training window = 6 months of rolling data
  - lookahead gap = 21 trading days between train end and prediction date
  - top_k = 2 (equal-weight the 2 highest predicted-return assets)
  - rebalance frequency = monthly (first trading day of each month)
  - cost model = Trading212 (15 bps commission + 3 bps slippage)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.neural_network import MLPRegressor

from utils import BACKTEST_START, RESULTS_ROOT, T212_COMMISSION, T212_SLIPPAGE, month_starts, run_multi_bucket
from utils.engine import run_benchmark
from utils.strategies import ml_signal


def run_single(close: pd.DataFrame, spx: pd.Series, results_dir: Path) -> None:
    """run mlp walk-forward for one bucket (trains on full history, backtests 2020+)."""
    rebal = month_starts(close)
    ws = ml_signal(
        close, rebal, top_k=2,
        model_factory=lambda: MLPRegressor(
            hidden_layer_sizes=(32, 16), max_iter=1000,
            early_stopping=True, random_state=42,
        ),
    )
    close_bt = close.loc[close.index >= BACKTEST_START]
    spx_bt = spx.loc[spx.index >= BACKTEST_START]
    ws_bt = {d: w for d, w in ws.items() if d >= BACKTEST_START}
    run_benchmark(ws_bt, close_bt, results_dir=results_dir,
                  title="08 ml-signal-mlp",
                  commission=T212_COMMISSION, slippage=T212_SLIPPAGE,
                  note="sklearn MLPRegressor walk-forward; trains on 2018-2020, backtests 2020-2025",
                  spx=spx_bt)


def main():
    """run across all buckets with full 7-year history for training."""
    run_multi_bucket(run_single, RESULTS_ROOT / "08-ml-signal" / "mlp",
                     n_assets=6, full_history=True)


if __name__ == "__main__":
    main()
