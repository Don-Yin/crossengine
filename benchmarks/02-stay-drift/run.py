"""02 -- stay-drift validation, all 5 engines.

tests the STAY signal (hold at drifted weight, do not rebalance). starts
with asset_a 60% / asset_b 40% on day 0. on day 60, asset_b is rebalanced
to 70% while asset_a keeps its drifted weight. all other days: both assets
hold (STAY). costs: 15 bps commission + 3 bps slippage.

runs across all stratified buckets, taking the first 2 tickers from each
bucket. this specifically tests how engines handle partial rebalancing where
one asset's weight is left to drift.

bt and vectorbt use custom adapters (bt reads drifted weight at runtime;
vectorbt pre-computes from price data). backtrader and cvxportfolio use
the generic WeightSchedule adapters with a pre-computed drifted weight.
"""

from __future__ import annotations

import logging
from pathlib import Path

import bt as _bt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from utils import (
    RESULTS_ROOT, T212_COMMISSION, T212_SLIPPAGE,
    run_bt, run_multi_bucket, run_vectorbt, write_comparison,
)
from utils.wrappers import run_backtrader_engine, run_cvxportfolio_engine

INITIAL_CASH = 100_000
REBAL_DAY = 60


# -- our engine ---------------------------------------------------------------

def run_engine(close, rebal_day, asset_a, asset_b):
    """run our engine with stay-drift logic."""
    from backtest import STAY, backtest

    rows = []
    for i in range(len(close)):
        if i == 0:
            rows.append({asset_a: 0.6, asset_b: 0.4})
        elif i == rebal_day:
            rows.append({asset_a: STAY, asset_b: 0.7})
        else:
            rows.append({asset_a: STAY, asset_b: STAY})
    return backtest(close, pd.DataFrame(rows, index=close.index),
                    initial_cash=INITIAL_CASH, commission=T212_COMMISSION,
                    slippage=T212_SLIPPAGE)


# -- bt reference (STAY = hold at drifted weight) ----------------------------

class _StayDriftBt(_bt.Algo):
    """bt algo that holds drifted weights except on rebalance day."""

    def __init__(self, day0, rebal, asset_a, asset_b):
        super().__init__()
        self.day0 = day0
        self.rebal = rebal
        self.asset_a = asset_a
        self.asset_b = asset_b

    def __call__(self, target):
        if target.now == self.day0:
            target.temp["weights"] = {self.asset_a: 0.6, self.asset_b: 0.4}
            return True
        if target.now == self.rebal:
            a_w = target.children[self.asset_a].value / target.value
            target.temp["weights"] = {self.asset_a: a_w, self.asset_b: 1.0 - a_w}
            return True
        return False


def _bt_ref(close, rebal_day, asset_a, asset_b):
    """run bt reference for stay-drift."""
    strategy = _bt.Strategy("stay-drift", [
        _StayDriftBt(close.index[0], close.index[rebal_day], asset_a, asset_b),
        _bt.algos.Rebalance(),
    ])
    return run_bt(close, strategy, initial_cash=INITIAL_CASH,
                  commission_rate=T212_COMMISSION + T212_SLIPPAGE)


# -- shared helpers -----------------------------------------------------------

def _drifted_weight(close, rebal_day, asset_a, asset_b) -> float:
    """pre-compute asset_a's drifted weight at rebal_day from price data."""
    pa, pb = close[asset_a].values, close[asset_b].values
    sa0 = INITIAL_CASH * 0.6 / pa[0]
    sb0 = INITIAL_CASH * 0.4 / pb[0]
    total_rd = sa0 * pa[rebal_day] + sb0 * pb[rebal_day]
    return sa0 * pa[rebal_day] / total_rd


def _stay_drift_ws(close, rebal_day, asset_a, asset_b) -> dict:
    """build a weight schedule for the stay-drift strategy."""
    a_w_rd = _drifted_weight(close, rebal_day, asset_a, asset_b)
    return {
        close.index[0]: {asset_a: 0.6, asset_b: 0.4},
        close.index[rebal_day]: {asset_a: a_w_rd, asset_b: 1.0 - a_w_rd},
    }


# -- vectorbt reference -------------------------------------------------------

def _vbt_ref(close, rebal_day, asset_a, asset_b):
    """run vectorbt reference for stay-drift."""
    a_w_rd = _drifted_weight(close, rebal_day, asset_a, asset_b)
    size = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    size.iloc[0] = [0.6, 0.4]
    size.iloc[rebal_day] = [a_w_rd, 1.0 - a_w_rd]
    return run_vectorbt(close, size, initial_cash=INITIAL_CASH,
                        commission_rate=T212_COMMISSION + T212_SLIPPAGE)


# -- backtrader reference -----------------------------------------------------

def _backtrader_ref(close, rebal_day, asset_a, asset_b):
    """run backtrader reference for stay-drift."""
    ws = _stay_drift_ws(close, rebal_day, asset_a, asset_b)
    return run_backtrader_engine(
        close, ws, initial_cash=INITIAL_CASH,
        commission=T212_COMMISSION + T212_SLIPPAGE,
    )


# -- cvxportfolio reference ---------------------------------------------------

def _cvxportfolio_ref(close, rebal_day, asset_a, asset_b):
    """run cvxportfolio reference for stay-drift."""
    ws = _stay_drift_ws(close, rebal_day, asset_a, asset_b)
    return run_cvxportfolio_engine(
        close, ws, initial_cash=INITIAL_CASH,
        commission=T212_COMMISSION + T212_SLIPPAGE,
    )


# -- main ---------------------------------------------------------------------

def run_single(close: pd.DataFrame, spx: pd.Series, results_dir: Path) -> None:
    """run stay-drift validation for one bucket pair."""
    asset_a, asset_b = close.columns[0], close.columns[1]
    logger.info("data: %d days, assets: %s, %s", len(close), asset_a, asset_b)
    logger.info("range: %s -- %s", close.index[0].date(), close.index[-1].date())
    logger.info("rebalance day: %d (%s)", REBAL_DAY, close.index[REBAL_DAY].date())

    from utils.data import compute_asset_avg
    asset_avg = compute_asset_avg(close, INITIAL_CASH)

    result = run_engine(close, REBAL_DAY, asset_a, asset_b)
    refs = {
        "bt": _bt_ref(close, REBAL_DAY, asset_a, asset_b),
        "vectorbt": _vbt_ref(close, REBAL_DAY, asset_a, asset_b),
        "backtrader": _backtrader_ref(close, REBAL_DAY, asset_a, asset_b),
        "cvxportfolio": _cvxportfolio_ref(close, REBAL_DAY, asset_a, asset_b),
    }
    write_comparison(result, refs, results_dir,
                     f"02 stay-drift ({asset_a}/{asset_b})",
                     note="all engines receive identical total cost rate",
                     spx=spx, asset_avg=asset_avg)


def main():
    """run across all stratified buckets."""
    run_multi_bucket(run_single, RESULTS_ROOT / "02-stay-drift", n_assets=2)


if __name__ == "__main__":
    main()
