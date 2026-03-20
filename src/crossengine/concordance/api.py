"""public API: concordance() function."""
from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from crossengine.concordance.engines import (
    detect_engines,
    run_backtrader_engine,
    run_bt_engine,
    run_cvxportfolio_engine,
    run_ours,
    run_vbt_engine,
)
from crossengine.concordance.report import ConcordanceReport
from crossengine.concordance.resolve import has_stay, resolve_stay
from crossengine.concordance.types import STAY, SignalSchedule, WeightSchedule

logger = logging.getLogger(__name__)


def _month_starts(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """first trading day of each month in the index."""
    firsts: set[pd.Timestamp] = set()
    seen_months: set[tuple[int, int]] = set()
    for d in index:
        key = (d.year, d.month)
        if key not in seen_months:
            seen_months.add(key)
            firsts.add(d)
    return firsts


def concordance(
    strategy: SignalSchedule | Callable[[pd.DataFrame, set[pd.Timestamp]], SignalSchedule],
    close: pd.DataFrame,
    *,
    rebal_dates: set[pd.Timestamp] | None = None,
    initial_cash: float = 100_000,
    commission: float = 0.0015,
    slippage: float = 0.0003,
    engines: tuple[str, ...] | None = None,
) -> ConcordanceReport:
    """run a strategy through multiple engines and measure concordance.

    accepts either a pre-computed SignalSchedule (dict) or a callable
    that produces one from (close, rebal_dates).
    """
    # resolve strategy to SignalSchedule
    if callable(strategy):
        dates = rebal_dates or _month_starts(close.index)
        ss: SignalSchedule = strategy(close, dates)
    else:
        ss = strategy

    if not ss:
        raise ValueError("empty signal schedule -- no rebalance dates produced")

    total_cost = commission + slippage

    # detect available engines
    available = detect_engines()
    requested = engines or tuple(available.keys())
    active = [e for e in requested if available.get(e, False)]

    if not active:
        raise RuntimeError("no engines available")

    logger.info("running %d engines: %s", len(active), ", ".join(active))

    # resolve STAY for category B engines
    need_resolved = has_stay(ss)
    ws_resolved: WeightSchedule = resolve_stay(ss, close, initial_cash, total_cost) if need_resolved else ss

    # dispatch
    equity: dict[str, pd.Series] = {}

    # category A: native STAY resolution (receive raw SignalSchedule)
    if "ours" in active:
        logger.info("running: ours")
        equity["ours"] = run_ours(close, ss, initial_cash=initial_cash, commission=commission, slippage=slippage)

    if "bt" in active:
        logger.info("running: bt")
        equity["bt"] = run_bt_engine(close, ss, initial_cash=initial_cash, commission=total_cost)

    if "backtrader" in active:
        logger.info("running: backtrader")
        equity["backtrader"] = run_backtrader_engine(close, ss, initial_cash=initial_cash, commission=total_cost)

    # category B: pre-resolved WeightSchedule (no STAY)
    if "vectorbt" in active:
        logger.info("running: vectorbt")
        equity["vectorbt"] = run_vbt_engine(close, ws_resolved, initial_cash=initial_cash, commission=total_cost)

    if "cvxportfolio" in active:
        logger.info("running: cvxportfolio")
        equity["cvxportfolio"] = run_cvxportfolio_engine(close, ws_resolved, initial_cash=initial_cash, commission=total_cost)

    return ConcordanceReport(equity, initial_cash)
