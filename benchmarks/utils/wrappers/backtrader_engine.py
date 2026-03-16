"""adapter for backtrader backtesting engine.

two critical configuration issues identified and forensically verified
(March 2026, against backtrader source code):

1. percabs 100x undercharge (definitive, comminfo.py:157):
   ``CommInfoBase.__init__`` divides commission by 100 when
   ``percabs=False`` (the default):
       if self._commtype == self.COMM_PERC and not self.p.percabs:
           self.p.commission /= 100.0
   passing commission=0.0018 (intent: 18 bps) applies 0.18 bps.
   documented in a cryptic docstring (lines 80-85) but contradicts
   every other engine's convention. the newer ``CommissionInfo``
   subclass defaults to ``percabs=True``, but ``CommInfoBase``
   retains the dangerous default.
   fix: set ``percabs=True`` in _FractionalCommInfo.

2. fill-ordering margin rejection (definitive, bbroker.py:558-583):
   ``check_submitted()`` pops orders from a FIFO deque and
   pseudo-executes each against a running cash balance. cash < 0
   triggers immediate margin rejection per-order. buy orders
   needing cash from later sells are rejected before those sells
   process. the FIFO order follows data-feed addition order.
   completely undocumented.
   fix: atomic delta computation, sells-first ordering, epsilon
   scale-down (1e-9) to absorb floating-point dust.

residual divergence after both fixes: 0.18-0.30%. the 0.18% floor
is equity-recording timing: backtrader records getvalue() before the
current bar's fills; our engine records after fills.
"""

from __future__ import annotations

import backtrader as _bt
import pandas as pd
from utils.types import WeightSchedule

_EPS = 1e-9


class _FractionalCommInfo(_bt.CommInfoBase):
    """fractional shares + correct commission rate (percabs=True)."""

    params = (
        ("commission", 0.0),
        ("stocklike", True),
        ("commtype", _bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
    )

    def getsize(self, price, cash):
        return cash / price

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * price * self.p.commission


class _AtomicRebalanceWS(_bt.Strategy):
    """atomic weight targeting with sell-before-buy ordering."""

    params = (("ws", {}), ("tickers", []), ("commission", 0.0))

    def __init__(self):
        self.vals, self.dts = [], []

    def next(self):
        dt = pd.Timestamp(self.datas[0].datetime.date(0))
        self.dts.append(dt)
        self.vals.append(self.broker.getvalue())
        weights = self.p.ws.get(dt)
        if weights is None:
            return

        pf_val = self.broker.getvalue()
        comm = self.p.commission

        sells, buys = [], []
        for i, d in enumerate(self.datas):
            t = self.p.tickers[i]
            price = d.close[0]
            target = pf_val * weights.get(t, 0.0) / price
            current = self.getposition(d).size
            delta = target - current
            if delta < -1e-10:
                sells.append((d, abs(delta)))
            elif delta > 1e-10:
                buys.append((d, delta, price))

        for d, size in sells:
            self.sell(d, size=size)

        if buys:
            sell_proceeds = sum(s * d.close[0] for d, s in sells)
            sell_comm = sum(s * d.close[0] * comm for d, s in sells)
            est_cash = self.broker.getcash() + sell_proceeds - sell_comm
            total_buy_cost = sum(sz * pr * (1 + comm) for _, sz, pr in buys)
            scale = min(1.0, est_cash / total_buy_cost) if total_buy_cost > 0 else 1.0
            scale *= 1 - _EPS
            for d, size, price in buys:
                self.buy(d, size=size * scale)


def run_backtrader_engine(
    close: pd.DataFrame,
    ws: WeightSchedule,
    *,
    initial_cash: float,
    commission: float,
) -> pd.Series:
    cerebro = _bt.Cerebro()
    tickers = close.columns.tolist()
    for col in tickers:
        ohlcv = pd.DataFrame(
            {
                "open": close[col],
                "high": close[col],
                "low": close[col],
                "close": close[col],
                "volume": 0,
            },
            index=close.index,
        )
        cerebro.adddata(_bt.feeds.PandasData(dataname=ohlcv), name=col)
    cerebro.addstrategy(
        _AtomicRebalanceWS,
        ws=ws,
        tickers=tickers,
        commission=commission,
    )
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.addcommissioninfo(_FractionalCommInfo(commission=commission))
    cerebro.broker.set_coc(True)
    strat = cerebro.run()[0]
    return pd.Series(strat.vals, index=strat.dts)
