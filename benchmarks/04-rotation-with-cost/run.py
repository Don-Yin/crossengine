"""04 -- large weight rotation (80/5/5/5/5) + 2x costs, vs bt + vectorbt.

identical strategy to benchmark 03 (80/5 rotation) but with doubled costs:
30 bps commission + 6 bps slippage. this isolates the effect of cost
magnitude on engine divergence. the same alternating 80/5/5/5/5 allocation
is applied monthly, but higher transaction costs amplify any differences in
how engines compute and apply fees.
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
    """run 80/5 rotation with 2x costs for one bucket's worth of assets."""
    tickers = close.columns.tolist()
    rebal = month_starts(close)
    ws = alternating_weights(tickers, rebal)
    run_benchmark(
        ws, close,
        results_dir=results_dir,
        title="04 large rotation + 2x costs",
        commission=T212_COMMISSION * 2,
        slippage=T212_SLIPPAGE * 2,
        note="2x Trading212 costs; all engines receive identical total cost rate",
        spx=spx,
    )


def main():
    run_multi_bucket(run_single, RESULTS_ROOT / "04-rotation-with-cost", n_assets=6)


if __name__ == "__main__":
    main()
