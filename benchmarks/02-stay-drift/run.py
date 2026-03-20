"""02 -- stay-drift validation, all 5 engines.

tests the STAY signal (hold at drifted weight, do not rebalance). starts
with asset_a 60% / asset_b 40% on day 0. on day 60, asset_b is rebalanced
to 70% while asset_a keeps its drifted weight. all other days: both assets
hold (STAY). costs: 15 bps commission + 3 bps slippage.

runs across all stratified buckets, taking the first 2 tickers from each
bucket. uses SignalSchedule with STAY -- all 5 engines go through the
unified run_benchmark() codepath. category A engines (ours, bt, backtrader)
resolve STAY at runtime; category B engines (vectorbt, cvxportfolio) receive
pre-resolved drifted weights via resolve_stay().
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from utils import RESULTS_ROOT, T212_COMMISSION, T212_SLIPPAGE, run_multi_bucket
from utils.engine import run_benchmark
from utils.types import STAY

logger = logging.getLogger(__name__)

REBAL_DAY = 60


def run_single(close: pd.DataFrame, spx: pd.Series, results_dir: Path) -> None:
    """run stay-drift validation for one bucket pair."""
    asset_a, asset_b = close.columns[0], close.columns[1]
    logger.info("data: %d days, assets: %s, %s", len(close), asset_a, asset_b)
    logger.info("range: %s -- %s", close.index[0].date(), close.index[-1].date())
    logger.info("rebalance day: %d (%s)", REBAL_DAY, close.index[REBAL_DAY].date())

    ss = {
        close.index[0]: {asset_a: 0.6, asset_b: 0.4},
        close.index[REBAL_DAY]: {asset_a: STAY, asset_b: 0.7},
    }

    run_benchmark(
        ss, close,
        results_dir=results_dir,
        title=f"02 stay-drift ({asset_a}/{asset_b})",
        commission=T212_COMMISSION,
        slippage=T212_SLIPPAGE,
        note="STAY signal: asset_a drifts, asset_b rebalanced to 70% on day 60",
        spx=spx,
    )


def main():
    """run across all stratified buckets."""
    run_multi_bucket(run_single, RESULTS_ROOT / "02-stay-drift", n_assets=2)


if __name__ == "__main__":
    main()
