"""01 -- equal-weight monthly rebalance, vs bt + vectorbt.

equal-weight allocation across all assets, rebalanced on the first trading
day of each calendar month. each asset receives weight = 1/N where N is
the number of assets. costs: 15 bps commission (Trading212 FX fee) + 3 bps
slippage. rebalance frequency: monthly. this is the simplest baseline
strategy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import RESULTS_ROOT, T212_COMMISSION, T212_SLIPPAGE, month_starts, run_multi_bucket
from utils.engine import run_benchmark


def run_single(close: pd.DataFrame, spx: pd.Series, results_dir: Path) -> None:
    """run equal-weight 1/N for one bucket's worth of assets."""
    tickers = close.columns.tolist()
    rebal = month_starts(close)
    ws = {d: {t: 1.0 / len(tickers) for t in tickers} for d in rebal}
    run_benchmark(
        ws, close,
        results_dir=results_dir,
        title="01 equal-weight monthly rebalance",
        commission=T212_COMMISSION,
        slippage=T212_SLIPPAGE,
        note="all engines receive identical total cost rate (commission + slippage)",
        spx=spx,
    )


def main():
    run_multi_bucket(run_single, RESULTS_ROOT / "01-equal-weight", n_assets=6)


if __name__ == "__main__":
    main()
