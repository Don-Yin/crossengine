"""adapter for our custom backtest engine.

status: primary reference engine. used as one of 5 engines in the
benchmark suite. the paper reports all C(5,2) = 10 pairwise
comparisons; our engine is not the sole arbiter but one data point.
to be published separately via JOSS companion paper.
"""
from __future__ import annotations

import pandas as pd
from utils.types import STAY as BENCH_STAY, SignalSchedule


def run_ours(
    close: pd.DataFrame,
    ss: SignalSchedule,
    *,
    initial_cash: float,
    commission: float,
    slippage: float,
):
    """run our engine with native STAY support."""
    from crossengine import STAY as ENGINE_STAY, backtest

    tickers = close.columns.tolist()
    rows = []
    for d in close.index:
        if d in ss:
            row = {}
            for t in tickers:
                sig = ss[d].get(t, 0.0)
                row[t] = ENGINE_STAY if sig == BENCH_STAY else sig
            rows.append(row)
        else:
            rows.append({t: ENGINE_STAY for t in tickers})
    signals = pd.DataFrame(rows, index=close.index)
    return backtest(
        close, signals,
        initial_cash=initial_cash,
        commission=commission,
        slippage=slippage,
    )
