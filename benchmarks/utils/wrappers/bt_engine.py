"""adapter for pmorissette/bt backtesting engine.

status: active. divergence range 0.00-1.13%.

divergence driver: r = 0.97 with total cost, r = 0.94 with
cost-per-trade (monotonic, predictable). zero-cost BM09: 0.000%.
divergence is purely cost-model implementation difference.

no bugs identified; the commission lambda correctly receives
quantity and price and applies proportional fee.
"""
from __future__ import annotations

import bt as _bt
import pandas as pd
from utils.types import STAY, SignalSchedule


class _SSAlgo(_bt.Algo):
    """bt algo that resolves STAY at runtime using live portfolio state."""

    def __init__(self, ss: SignalSchedule, tickers: list[str]):
        super().__init__()
        self._ss = ss
        self._tickers = tickers

    def __call__(self, target):
        if target.now not in self._ss:
            return False

        raw = self._ss[target.now]
        weights = {}
        stay_value = 0.0
        pf_value = target.value

        # phase 1: compute STAY assets' drifted value
        for t in self._tickers:
            sig = raw.get(t, 0.0)
            if sig == STAY and t in target.children and pf_value > 0:
                stay_value += target.children[t].value

        # phase 2: allocate active assets from remaining budget
        budget_frac = max(0.0, 1.0 - stay_value / pf_value) if pf_value > 0 else 0.0
        active_sum = sum(
            abs(float(raw.get(t, 0.0)))
            for t in self._tickers
            if raw.get(t, 0.0) != STAY
        )

        for t in self._tickers:
            sig = raw.get(t, 0.0)
            if sig == STAY:
                child_val = target.children[t].value if t in target.children else 0.0
                weights[t] = child_val / pf_value if pf_value > 0 else 0.0
            elif active_sum > 0:
                weights[t] = abs(float(sig)) / active_sum * budget_frac
            else:
                weights[t] = 0.0

        target.temp["weights"] = weights
        return True


def run_bt_engine(
    close: pd.DataFrame,
    ss: SignalSchedule,
    *,
    name: str,
    initial_cash: float,
    commission: float,
) -> pd.Series:
    """run bt engine with native STAY support."""
    comm_fn = (lambda q, p: abs(q) * p * commission) if commission else (lambda q, p: 0)
    tickers = close.columns.tolist()
    strategy = _bt.Strategy(name, [_SSAlgo(ss, tickers), _bt.algos.Rebalance()])
    test = _bt.Backtest(
        strategy, close,
        initial_capital=initial_cash,
        commissions=comm_fn,
        integer_positions=False,
    )
    res = _bt.run(test)
    key = list(res.keys())[0]
    rebased = res.prices[key]
    return rebased / rebased.iloc[0] * initial_cash
