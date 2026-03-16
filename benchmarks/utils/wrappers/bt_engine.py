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
from utils.types import WeightSchedule


class _WSAlgo(_bt.Algo):
    def __init__(self, ws: WeightSchedule, tickers: list[str]):
        super().__init__()
        self._ws = ws
        self._tickers = tickers

    def __call__(self, target):
        if target.now in self._ws:
            raw = self._ws[target.now]
            target.temp["weights"] = {t: raw.get(t, 0.0) for t in self._tickers}
            return True
        return False


def run_bt_engine(
    close: pd.DataFrame,
    ws: WeightSchedule,
    *,
    name: str,
    initial_cash: float,
    commission: float,
) -> pd.Series:
    comm_fn = (lambda q, p: abs(q) * p * commission) if commission else (lambda q, p: 0)
    tickers = close.columns.tolist()
    strategy = _bt.Strategy(name, [_WSAlgo(ws, tickers), _bt.algos.Rebalance()])
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
