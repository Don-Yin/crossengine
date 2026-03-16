"""03 -- large weight rotation (80/5/5/5/5), vs bt + vectorbt.

large weight rotation, alternating between two extreme allocations each
month. on odd months: 80% in the first asset, 5% in each of the remaining
four. on even months: 80% in the last asset, 5% in the rest. rebalance
frequency: monthly (first trading day). costs: 15 bps commission + 3 bps
slippage. this generates large turnover to stress-test cost computation and
order-of-execution handling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import (
    RESULTS_ROOT, T212_COMMISSION, T212_SLIPPAGE,
    alternating_weights, month_starts, run_multi_bucket,
)
from utils.engine import run_benchmark


def run_single(close: pd.DataFrame, spx: pd.Series, results_dir: Path) -> None:
    """run 80/5 rotation for one bucket's worth of assets."""
    tickers = close.columns.tolist()
    rebal = month_starts(close)
    ws = alternating_weights(tickers, rebal)
    run_benchmark(
        ws, close,
        results_dir=results_dir,
        title="03 large rotation",
        commission=T212_COMMISSION,
        slippage=T212_SLIPPAGE,
        note="all engines receive identical total cost rate",
        spx=spx,
    )


def main():
    run_multi_bucket(run_single, RESULTS_ROOT / "03-rotation", n_assets=6)


if __name__ == "__main__":
    main()
