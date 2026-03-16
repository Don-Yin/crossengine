"""adapter for NautilusTrader backtesting engine (excluded).

status: excluded due to 4 compounding issues, 3 definitive
(verified March 2026, against Rust source code).

1. HEDGING phantom positions (definitive, ids_generator.rs):
   OmsType.HEDGING generates a new position ID per order; selling
   to reduce a LONG creates a separate SHORT. documented intended
   behavior for multi-strategy desks, but a configuration trap for
   single-strategy backtesting -- docs don't warn against this.

2. NETTING silent truncation (probable):
   engine processed 62/1825 days then terminated. likely a Rust
   panic during position-flip PnL calculation (GitHub issues #1512,
   #963 document related failures). subprocess wrapper swallows
   the crash.

3. double commission (definitive, fee.rs + _rebalance below):
   wrapper reduces target qty by 1/(1+c) to reserve cash for fees,
   but engine also charges MakerTakerFeeModel fees on every fill.
   commission applied twice.

4. Rust Money overflow (definitive, money.rs:482-497):
   Money::Add uses checked_add().expect(); overflow triggers
   panic!() -> SIGABRT. MONEY_MAX of ~$9.2B can be exceeded by
   intermediate calculations during intensive daily rebalancing.

the wrapper runs NautilusTrader in a forked subprocess to isolate
Rust panics from the main process (SIGABRT would kill the parent).
"""
from __future__ import annotations

import logging
from decimal import Decimal

import pandas as pd
from utils.types import WeightSchedule

logger = logging.getLogger(__name__)


def run_nautilus_engine(
    close: pd.DataFrame,
    ws: WeightSchedule,
    *,
    initial_cash: float,
    commission: float,
) -> pd.Series:
    """run nautilus_trader in a forked subprocess to isolate Rust panics."""
    import multiprocessing as mp
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mktemp(suffix=".csv", prefix="naut_eq_"))
    ctx = mp.get_context("fork")
    p = ctx.Process(
        target=_subprocess_run,
        args=(close, ws, initial_cash, commission, str(tmp)),
    )
    p.start()
    p.join(timeout=120)
    if p.is_alive():
        p.kill()
        p.join()
    if p.exitcode != 0 or not tmp.exists():
        logger.warning("nautilus subprocess exited with code %s", p.exitcode)
        tmp.unlink(missing_ok=True)
        return pd.Series(float("nan"), index=close.index)
    result = pd.read_csv(str(tmp), index_col=0, parse_dates=True).iloc[:, 0]
    tmp.unlink(missing_ok=True)
    return result


def _subprocess_run(close, ws, initial_cash, commission, out_path):
    """entry point for the forked subprocess."""
    try:
        result = _run(close, ws, initial_cash=initial_cash, commission=commission)
        result.to_csv(out_path)
    except Exception as exc:
        logger.warning("nautilus _run failed: %s", exc)


def _run(
    close: pd.DataFrame,
    ws: WeightSchedule,
    *,
    initial_cash: float,
    commission: float,
) -> pd.Series:
    """full NautilusTrader BacktestEngine pipeline."""
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
    from nautilus_trader.model.instruments.equity import Equity
    from nautilus_trader.model.objects import Money, Price, Quantity

    tickers = close.columns.tolist()
    venue = Venue("SIM")
    fee = Decimal(str(commission))

    engine = BacktestEngine(config=BacktestEngineConfig(trader_id="BACKTESTER-001"))
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.HEDGING,
        account_type=AccountType.CASH,
        starting_balances=[Money(initial_cash, USD)],
        base_currency=USD,
        allow_cash_borrowing=True,
    )

    for t in tickers:
        engine.add_instrument(Equity(
            instrument_id=InstrumentId(Symbol(t), venue),
            raw_symbol=Symbol(t),
            currency=USD,
            price_precision=2,
            price_increment=Price.from_str("0.01"),
            lot_size=Quantity.from_int(1),
            maker_fee=fee,
            taker_fee=fee,
            ts_event=0,
            ts_init=0,
        ))

    for t in tickers:
        bar_type = BarType.from_str(f"{t}.SIM-1-DAY-LAST-EXTERNAL")
        bars = []
        for dt, px in zip(close.index, close[t].values):
            ts_ns = int(pd.Timestamp(dt).timestamp() * 1_000_000_000)
            bars.append(Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{px:.2f}"),
                high=Price.from_str(f"{px:.2f}"),
                low=Price.from_str(f"{px:.2f}"),
                close=Price.from_str(f"{px:.2f}"),
                volume=Quantity.from_str("100000"),
                ts_event=ts_ns,
                ts_init=ts_ns,
            ))
        engine.add_data(bars)

    from nautilus_trader.model.enums import OrderSide, TimeInForce
    from nautilus_trader.trading.strategy import Strategy

    class _Rebalancer(Strategy):
        """rebalances portfolio on WeightSchedule dates."""

        def __init__(self):
            super().__init__()
            self._tickers = tickers
            self._ws = ws
            self._commission = commission
            self._venue = venue
            self._latest_prices: dict[str, float] = {}
            self._bars_this_ts: dict[pd.Timestamp, set[str]] = {}
            self.equity_curve: list[tuple[pd.Timestamp, float]] = []

        def on_start(self):
            for t in self._tickers:
                self.subscribe_bars(BarType.from_str(f"{t}.SIM-1-DAY-LAST-EXTERNAL"))

        def on_stop(self):
            pass

        def on_bar(self, bar):
            ticker = bar.bar_type.instrument_id.symbol.value
            ts = pd.Timestamp(bar.ts_event, unit="ns")
            self._latest_prices[ticker] = float(bar.close)

            if ts not in self._bars_this_ts:
                self._bars_this_ts[ts] = set()
            self._bars_this_ts[ts].add(ticker)
            if len(self._bars_this_ts[ts]) < len(self._tickers):
                return

            from nautilus_trader.model.currencies import USD
            total = self._portfolio_value(USD)
            self.equity_curve.append((ts, total))
            weights = self._ws.get(ts)
            if weights is not None:
                self._rebalance(weights, total)

        def _portfolio_value(self, usd):
            account = self.portfolio.account(self._venue)
            cash = float(account.balance_total(usd)) if account else 0.0
            pos_val = sum(
                float(pos.quantity) * self._latest_prices.get(t, 0.0)
                for t in self._tickers
                for pos in self.cache.positions(
                    instrument_id=InstrumentId(Symbol(t), self._venue),
                )
                if pos.is_open
            )
            return cash + pos_val

        def _rebalance(self, weights: dict[str, float], total: float):
            for t in self._tickers:
                target_w = weights.get(t, 0.0)
                price = self._latest_prices[t]
                target_qty = int(total * target_w / (price * (1 + self._commission)))
                current_qty = self._position_qty(t)
                diff = target_qty - current_qty
                if abs(diff) < 1:
                    continue
                side = OrderSide.BUY if diff > 0 else OrderSide.SELL
                self.submit_order(self.order_factory.market(
                    instrument_id=InstrumentId(Symbol(t), self._venue),
                    order_side=side,
                    quantity=Quantity.from_int(abs(diff)),
                    time_in_force=TimeInForce.GTC,
                ))

        def _position_qty(self, ticker: str) -> int:
            iid = InstrumentId(Symbol(ticker), self._venue)
            for pos in self.cache.positions(instrument_id=iid):
                if pos.is_open:
                    return int(pos.quantity) if pos.side.name == "LONG" else -int(pos.quantity)
            return 0

    strategy = _Rebalancer()
    engine.add_strategy(strategy)
    engine.run()

    result = pd.Series(dict(strategy.equity_curve), dtype=float).sort_index()
    engine.dispose()
    return result


