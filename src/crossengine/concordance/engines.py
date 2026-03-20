"""engine runners with graceful import detection."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from crossengine.concordance.types import STAY, SignalSchedule, WeightSchedule

logger = logging.getLogger(__name__)


def _check_engine(name: str) -> bool:
    """check if an engine package is importable."""
    import importlib
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def detect_engines() -> dict[str, bool]:
    """detect which engine packages are installed."""
    return {
        "ours": True,
        "bt": _check_engine("bt"),
        "vectorbt": _check_engine("vectorbt"),
        "backtrader": _check_engine("backtrader"),
        "cvxportfolio": _check_engine("cvxportfolio"),
    }


def run_ours(
    close: pd.DataFrame,
    ss: SignalSchedule,
    *,
    initial_cash: float,
    commission: float,
    slippage: float,
) -> pd.Series:
    """run our engine with native STAY support."""
    from crossengine import STAY as ENGINE_STAY, backtest

    tickers = close.columns.tolist()
    rows = []
    for d in close.index:
        if d in ss:
            row = {}
            for t in tickers:
                sig = ss[d].get(t, 0.0)
                row[t] = ENGINE_STAY if sig == STAY else sig
            rows.append(row)
        else:
            rows.append({t: ENGINE_STAY for t in tickers})
    signals = pd.DataFrame(rows, index=close.index)
    result = backtest(close, signals, initial_cash=initial_cash, commission=commission, slippage=slippage)
    return result.portfolio_value


def run_bt_engine(
    close: pd.DataFrame,
    ss: SignalSchedule,
    *,
    initial_cash: float,
    commission: float,
) -> pd.Series:
    """run bt engine with native STAY support."""
    import bt as _bt

    tickers = close.columns.tolist()

    class _SSAlgo(_bt.Algo):
        def __call__(self_algo, target):
            if target.now not in ss:
                return False
            raw = ss[target.now]
            weights = {}
            stay_value = 0.0
            pf_value = target.value
            for t in tickers:
                sig = raw.get(t, 0.0)
                if sig == STAY and t in target.children and pf_value > 0:
                    stay_value += target.children[t].value
            budget_frac = max(0.0, 1.0 - stay_value / pf_value) if pf_value > 0 else 0.0
            active_sum = sum(abs(float(raw.get(t, 0.0))) for t in tickers if raw.get(t, 0.0) != STAY)
            for t in tickers:
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

    comm_fn = (lambda q, p: abs(q) * p * commission) if commission else (lambda q, p: 0)
    strategy = _bt.Strategy("concordance", [_SSAlgo(), _bt.algos.Rebalance()])
    test = _bt.Backtest(strategy, close, initial_capital=initial_cash, commissions=comm_fn, integer_positions=False)
    res = _bt.run(test)
    key = list(res.keys())[0]
    rebased = res.prices[key]
    return rebased / rebased.iloc[0] * initial_cash


def run_vbt_engine(
    close: pd.DataFrame,
    ws: WeightSchedule,
    *,
    initial_cash: float,
    commission: float,
) -> pd.Series:
    """run vectorbt engine (receives pre-resolved weights)."""
    import vectorbt as vbt

    tickers = close.columns.tolist()
    size = pd.DataFrame(np.nan, index=close.index, columns=tickers)
    for d, weights in ws.items():
        if d in size.index:
            for t in tickers:
                size.loc[d, t] = weights.get(t, 0.0)

    pf = vbt.Portfolio.from_orders(
        close, size, size_type="targetpercent", fees=commission,
        init_cash=initial_cash, freq="1D", group_by=True, cash_sharing=True, call_seq="auto",
    )
    return pf.value()


def run_backtrader_engine(
    close: pd.DataFrame,
    ss: SignalSchedule,
    *,
    initial_cash: float,
    commission: float,
) -> pd.Series:
    """run backtrader engine with native STAY support."""
    import backtrader as _bt

    tickers = close.columns.tolist()

    class _FractionalCommInfo(_bt.CommInfoBase):
        params = (("commission", 0.0), ("stocklike", True), ("commtype", _bt.CommInfoBase.COMM_PERC), ("percabs", True))
        def getsize(self, price, cash): return cash / price
        def _getcommission(self, size, price, pseudoexec): return abs(size) * price * self.p.commission

    class _Strategy(_bt.Strategy):
        params = (("ss", {}), ("tickers", []), ("commission", 0.0))
        def __init__(self): self.vals, self.dts = [], []
        def next(self):
            dt = pd.Timestamp(self.datas[0].datetime.date(0))
            self.dts.append(dt)
            self.vals.append(self.broker.getvalue())
            signals = self.p.ss.get(dt)
            if signals is None:
                return
            pf_val = self.broker.getvalue()
            comm = self.p.commission
            stay_value, stay_assets = 0.0, set()
            for i, d in enumerate(self.datas):
                t = self.p.tickers[i]
                if signals.get(t, 0.0) == STAY:
                    stay_assets.add(t)
                    stay_value += self.getposition(d).size * d.close[0]
            budget = pf_val - stay_value
            active_sigs = {t: float(signals.get(t, 0.0)) for t in self.p.tickers if t not in stay_assets}
            active_sum = sum(abs(v) for v in active_sigs.values())
            sells, buys = [], []
            for i, d in enumerate(self.datas):
                t = self.p.tickers[i]
                if t in stay_assets:
                    continue
                price = d.close[0]
                target = budget * abs(active_sigs.get(t, 0.0)) / active_sum / price if active_sum > 0 and budget > 0 else 0.0
                current = self.getposition(d).size
                delta = target - current
                if delta < -1e-10: sells.append((d, abs(delta)))
                elif delta > 1e-10: buys.append((d, delta, price))
            for d, size in sells: self.sell(d, size=size)
            if buys:
                sell_proceeds = sum(s * d.close[0] for d, s in sells)
                sell_comm = sum(s * d.close[0] * comm for d, s in sells)
                est_cash = self.broker.getcash() + sell_proceeds - sell_comm
                total_buy_cost = sum(sz * pr * (1 + comm) for _, sz, pr in buys)
                scale = min(1.0, est_cash / total_buy_cost) if total_buy_cost > 0 else 1.0
                scale *= 1 - 1e-9
                for d, size, price in buys: self.buy(d, size=size * scale)

    cerebro = _bt.Cerebro()
    for col in tickers:
        ohlcv = pd.DataFrame({"open": close[col], "high": close[col], "low": close[col], "close": close[col], "volume": 0}, index=close.index)
        cerebro.adddata(_bt.feeds.PandasData(dataname=ohlcv), name=col)
    cerebro.addstrategy(_Strategy, ss=ss, tickers=tickers, commission=commission)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.addcommissioninfo(_FractionalCommInfo(commission=commission))
    cerebro.broker.set_coc(True)
    strat = cerebro.run()[0]
    return pd.Series(strat.vals, index=strat.dts)


def run_cvxportfolio_engine(
    close: pd.DataFrame,
    ws: WeightSchedule,
    *,
    initial_cash: float,
    commission: float,
) -> pd.Series:
    """run cvxportfolio engine (receives pre-resolved weights)."""
    import cvxportfolio as cvx

    tickers = close.columns.tolist()
    returns_fwd = close.pct_change().shift(-1)
    market_data = cvx.UserProvidedMarketData(returns=returns_fwd, prices=close, cash_key="cash", min_history=pd.Timedelta("0d"))
    costs = [cvx.TransactionCost(a=commission, b=None)] if commission else []
    sim = cvx.MarketSimulator(market_data=market_data, costs=costs, round_trades=False, cash_key="cash")
    rows, dates = [], []
    for dt, weights in sorted(ws.items()):
        row = {t: weights.get(t, 0.0) for t in tickers}
        row["cash"] = max(0.0, 1.0 - sum(row.values()))
        rows.append(row)
        dates.append(dt)
    target_df = pd.DataFrame(rows, index=dates)
    policy = cvx.FixedWeights(target_df)
    result = sim.backtest(policy, start_time=close.index[0], end_time=close.index[-1], initial_value=initial_cash)
    return result.v
