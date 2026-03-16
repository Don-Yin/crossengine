"""adapter for vectorbt backtesting engine.

status: active. divergence range 0.00-0.08% (closest to ours).

divergence driver: r = 0.75 with ML signal complexity, -0.70 with
trade count (orthogonal to bt's cost-driven divergence). zero-cost
BM09: 0.000%. no bugs identified; ``from_orders`` with
``targetpercent`` + ``cash_sharing`` + ``call_seq='auto'``
correctly handles multi-asset rebalancing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from utils.types import WeightSchedule


def run_vbt_engine(
    close: pd.DataFrame,
    ws: WeightSchedule,
    *,
    initial_cash: float,
    commission: float,
) -> pd.Series:
    import vectorbt as vbt

    tickers = close.columns.tolist()
    size = pd.DataFrame(np.nan, index=close.index, columns=tickers)
    for d, weights in ws.items():
        if d in size.index:
            for t in tickers:
                size.loc[d, t] = weights.get(t, 0.0)

    pf = vbt.Portfolio.from_orders(
        close, size,
        size_type="targetpercent",
        fees=commission,
        init_cash=initial_cash,
        freq="1D",
        group_by=True,
        cash_sharing=True,
        call_seq="auto",
    )
    return pf.value()
