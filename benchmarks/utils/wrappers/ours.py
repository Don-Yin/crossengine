"""adapter for our custom backtest engine.

status: primary reference engine. used as one of 5 engines in the
benchmark suite. the paper reports all C(5,2) = 10 pairwise
comparisons; our engine is not the sole arbiter but one data point.
to be published separately via JOSS companion paper.
"""
from __future__ import annotations

import pandas as pd
from utils.types import WeightSchedule


def run_ours(
    close: pd.DataFrame,
    ws: WeightSchedule,
    *,
    initial_cash: float,
    commission: float,
    slippage: float,
):
    from backtest import STAY, backtest

    tickers = close.columns.tolist()
    rows = []
    for d in close.index:
        if d in ws:
            rows.append({t: ws[d].get(t, 0.0) for t in tickers})
        else:
            rows.append({t: STAY for t in tickers})
    signals = pd.DataFrame(rows, index=close.index)
    return backtest(
        close, signals,
        initial_cash=initial_cash,
        commission=commission,
        slippage=slippage,
    )
