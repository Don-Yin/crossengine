"""Core backtest engine.

The public entry point is :func:`backtest`.  It accepts a prices DataFrame (or
:class:`~backtest.data.OHLCV`), a signals DataFrame, and optional order /
commission / slippage configuration -- then returns a :class:`BacktestResult`.
"""

from __future__ import annotations

import pandas as pd

from .data import OHLCV
from .models.commission import CommissionModel, make_commission
from .models.slippage import SlippageModel, make_slippage
from .orders import Order, OrderQueue, OrderType, check_pending_fill
from .portfolio import Portfolio
from .result import BacktestResult
from .signals import STAY, resolve_signals


def backtest(
    prices: pd.DataFrame | OHLCV,
    signals: pd.DataFrame,
    *,
    orders: OrderQueue | None = None,
    initial_cash: float = 10_000,
    commission: float | CommissionModel = 0.001,
    slippage: float | SlippageModel = 0.0,
    long_only: bool = True,
    risk_free: float = 0.04,
) -> BacktestResult:
    """Run a multi-asset portfolio backtest.

    Parameters
    ----------
    prices
        Asset prices.  A plain ``DataFrame`` is treated as close prices
        (index = dates, columns = asset names).  Pass an :class:`OHLCV` when
        the backtest needs high/low (for limit / stop order simulation).
    signals
        Target allocation signals — same column names as *prices*.
        Values are either **numeric** (interpreted as relative weights that
        will be auto-normalised) or the string ``"s"`` (:data:`STAY`) which
        means "keep the current number of shares -- do not trade this asset".
        Rows that exist in *signals* trigger a rebalance on that date;
        dates without a row simply hold.
    orders
        Optional queue of limit / stop orders.  When an order's
        ``submit_date`` matches the current bar **and** the corresponding
        asset has a non-STAY signal, the engine queues the order instead of
        executing a market fill.  The order then checks against subsequent
        bars' high/low for a fill.
    initial_cash
        Starting cash balance.
    commission
        A ``float`` is interpreted as a proportional rate (e.g. ``0.001`` =
        0.1 %).  Or pass any object with a ``.compute(qty, price)`` method.
    slippage
        A ``float`` is interpreted as a fixed proportional slippage.  Or pass
        any object with an ``.apply(price, qty, volume)`` method.
    long_only
        Clip negative signals to zero.
    risk_free
        Annual risk-free rate used for Sharpe / Sortino / Omega.

    Returns
    -------
    BacktestResult
    """
    data = _coerce_prices(prices)
    comm = make_commission(commission)
    slip = make_slippage(slippage)
    oq = orders or OrderQueue()

    signals = _align_signals(signals, data)

    engine = _Engine(data, signals, oq, initial_cash, comm, slip, long_only)
    chronicle, trade_log = engine.run()

    chronicle_df = pd.DataFrame(chronicle).set_index("date")
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(columns=["date", "asset", "side", "quantity", "price", "commission", "slippage", "type", "tag"])
    return BacktestResult(chronicle_df, trades_df, risk_free)


# ---------------------------------------------------------------------------
# Internal engine
# ---------------------------------------------------------------------------


class _Engine:
    def __init__(
        self,
        data: OHLCV,
        signals: pd.DataFrame,
        order_queue: OrderQueue,
        initial_cash: float,
        commission: CommissionModel,
        slippage: SlippageModel,
        long_only: bool,
    ) -> None:
        self.data = data
        self.signals = signals
        self.oq = order_queue
        self.comm = commission
        self.slip = slippage
        self.long_only = long_only

        self.portfolio = Portfolio(initial_cash, data.assets)
        self.pending: list[Order] = []

    # -- main loop ----------------------------------------------------------

    # TODO(execution_delay): add execution_delay parameter to backtest().
    # When execution_delay=1, buffer today's signal and execute at the next
    # bar's open price instead of the current bar's close.  This eliminates
    # look-ahead bias from close-price fills on same-bar signals.  Requires
    # OHLCV.open to be populated; fall back to close if open is None.

    def run(self) -> tuple[list[dict], list[dict]]:
        chronicle: list[dict] = []
        trade_log: list[dict] = []

        for i, date in enumerate(self.data.dates):
            bar = self._bar(i)
            self._process_pending(bar, date, trade_log)

            if date in self.signals.index:
                self._rebalance(bar, date, trade_log)

            chronicle.append(self.portfolio.snapshot(bar["prices"], date))

        return chronicle, trade_log

    # -- bar helpers --------------------------------------------------------

    def _bar(self, i: int) -> dict:
        d = self.data
        bar: dict = {
            "prices": {a: float(d.close[a].iloc[i]) for a in self.portfolio.assets},
        }
        if d.high is not None:
            bar["high"] = {a: float(d.high[a].iloc[i]) for a in self.portfolio.assets}
        if d.low is not None:
            bar["low"] = {a: float(d.low[a].iloc[i]) for a in self.portfolio.assets}
        if d.volume is not None:
            bar["volume"] = {a: float(d.volume[a].iloc[i]) for a in self.portfolio.assets}
        return bar

    # -- pending order processing -------------------------------------------

    def _process_pending(self, bar: dict, date: pd.Timestamp, trade_log: list[dict]) -> None:
        still_pending: list[Order] = []
        for order in self.pending:
            h = bar.get("high", {}).get(order.asset)
            l = bar.get("low", {}).get(order.asset)
            fill_price = check_pending_fill(order, h, l, bar["prices"][order.asset], date)

            if fill_price is not None:
                self._fill(order.asset, order.side, order.quantity, fill_price, bar, trade_log, date, order.order_type.value, order.tag)
            elif order.valid_until is not None and date > order.valid_until:
                pass  # expired
            else:
                still_pending.append(order)

        self.pending = still_pending

    # -- signal-driven rebalancing ------------------------------------------

    def _rebalance(self, bar: dict, date: pd.Timestamp, trade_log: list[dict]) -> None:
        row = self.signals.loc[date]

        raw: dict[str, float | str] = {}
        for a in self.portfolio.assets:
            raw[a] = row[a] if a in row.index else STAY

        if all(v == STAY for v in raw.values()):
            return

        prices = bar["prices"]
        total_value = self.portfolio.total_value(prices)

        target_shares = resolve_signals(
            raw,
            self.portfolio.positions,
            prices,
            total_value,
            self.long_only,
        )

        date_orders = self.oq.on_date(date)
        order_assets = {o.asset for o in date_orders}

        sells: dict[str, float] = {}
        buys: dict[str, float] = {}
        for a in self.portfolio.assets:
            delta = target_shares.get(a, 0.0) - self.portfolio.shares(a)
            if abs(delta) * prices.get(a, 0.0) < 0.01:
                continue
            if a in order_assets:
                for o in date_orders:
                    if o.asset == a:
                        o.quantity = abs(delta)
                        o.side = "buy" if delta > 0 else "sell"
                        self.pending.append(o)
            elif delta < 0:
                sells[a] = delta
            else:
                buys[a] = delta

        for a, delta in sells.items():
            self._fill(a, "sell", abs(delta), prices[a], bar, trade_log, date, "market", "")

        if buys:
            buy_costs: dict[str, tuple[float, float, float]] = {}
            total_cost = 0.0
            for a, delta in buys.items():
                qty = abs(delta)
                vol = bar.get("volume", {}).get(a)
                ep = self.slip.apply(prices[a], qty, vol)
                c = self.comm.compute(qty, ep)
                buy_costs[a] = (qty, ep, c)
                total_cost += qty * ep + c

            scale = min(self.portfolio.cash / total_cost, 1.0) if total_cost > 0 else 0.0

            for a, (qty, ep, _) in buy_costs.items():
                adj_qty = qty * scale
                if adj_qty < 1e-10:
                    continue
                self._fill(a, "buy", adj_qty, prices[a], bar, trade_log, date, "market", "")

    # -- single fill --------------------------------------------------------

    def _fill(
        self,
        asset: str,
        side: str,
        qty: float,
        ref_price: float,
        bar: dict,
        trade_log: list[dict],
        date: pd.Timestamp,
        order_type: str,
        tag: str,
    ) -> None:
        vol = bar.get("volume", {}).get(asset)
        signed_qty = qty if side == "buy" else -qty
        ep = self.slip.apply(ref_price, signed_qty, vol)
        c = self.comm.compute(qty, ep)

        self.portfolio.fill(asset, side, qty, ep, c)

        slip_cost = abs(ep - ref_price) * qty

        trade_log.append(
            {
                "date": date,
                "asset": asset,
                "side": side,
                "quantity": round(qty, 8),
                "price": round(ep, 6),
                "commission": round(c, 6),
                "slippage": round(slip_cost, 6),
                "type": order_type,
                "tag": tag,
            }
        )


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------


def _coerce_prices(prices: pd.DataFrame | OHLCV) -> OHLCV:
    if isinstance(prices, OHLCV):
        return prices
    if isinstance(prices, pd.DataFrame):
        return OHLCV(close=prices)
    raise TypeError(f"prices must be a DataFrame or OHLCV, got {type(prices)}")


def _align_signals(signals: pd.DataFrame, data: OHLCV) -> pd.DataFrame:
    """Ensure signals index is DatetimeIndex."""
    sig = signals.copy()
    if not isinstance(sig.index, pd.DatetimeIndex):
        sig.index = pd.to_datetime(sig.index)
    return sig
