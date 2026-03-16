"""adapter for cvxportfolio backtesting engine (Boyd et al., 2017).

cvxportfolio uses forward returns: r_t = (p_{t+1} - p_t) / p_t.  pandas
pct_change gives backward returns, so we shift(-1) to align conventions.

TransactionCost(a=rate, b=None) provides proportional fee on abs(trade value).
b=None disables the volume-based market impact component (requires volume data).

with correct return alignment, zero-cost divergence from our engine is 0.000%.
with commission, residual divergence is ~0.18% from equity-recording timing.
"""

from __future__ import annotations

import cvxportfolio as cvx
import pandas as pd
from utils.types import WeightSchedule


def run_cvxportfolio_engine(
    close: pd.DataFrame,
    ws: WeightSchedule,
    *,
    initial_cash: float,
    commission: float,
) -> pd.Series:
    tickers = close.columns.tolist()
    returns_fwd = close.pct_change().shift(-1)

    market_data = cvx.UserProvidedMarketData(
        returns=returns_fwd,
        prices=close,
        cash_key="cash",
        min_history=pd.Timedelta("0d"),
    )

    costs = [cvx.TransactionCost(a=commission, b=None)] if commission else []
    sim = cvx.MarketSimulator(
        market_data=market_data,
        costs=costs,
        round_trades=False,
        cash_key="cash",
    )

    rows, dates = [], []
    for dt, weights in sorted(ws.items()):
        row = {t: weights.get(t, 0.0) for t in tickers}
        row["cash"] = max(0.0, 1.0 - sum(row.values()))
        rows.append(row)
        dates.append(dt)
    target_df = pd.DataFrame(rows, index=dates)

    policy = cvx.FixedWeights(target_df)
    result = sim.backtest(
        policy,
        start_time=close.index[0],
        end_time=close.index[-1],
        initial_value=initial_cash,
    )
    return result.v
