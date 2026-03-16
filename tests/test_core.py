"""Smoke tests for the backtest engine."""

import numpy as np
import pandas as pd

from backtest import OHLCV, STAY, IBKRTiered, Order, OrderQueue, OrderType, backtest


def _make_prices(n_days: int = 20) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    return pd.DataFrame({
        "AAPL": 150 + np.cumsum(np.random.randn(n_days) * 1.5),
        "MSFT": 310 + np.cumsum(np.random.randn(n_days) * 2.0),
        "GOOGL": 140 + np.cumsum(np.random.randn(n_days) * 1.8),
    }, index=dates)


def test_basic_equal_weight():
    prices = _make_prices()
    signals = pd.DataFrame({
        "AAPL": [0.33] * len(prices),
        "MSFT": [0.33] * len(prices),
        "GOOGL": [0.34] * len(prices),
    }, index=prices.index)

    result = backtest(prices, signals, initial_cash=10_000, commission=0.001)
    assert result.portfolio_value.iloc[0] > 0
    assert len(result.trades) > 0
    m = result.metrics
    assert "sharpe_ratio" in m
    assert "total_return_pct" in m
    print(f"  equal weight: {result}")


def test_stay_freezes_shares_not_weights():
    """STAY must freeze share count.  Even if prices move, no trades should
    be generated for STAY assets."""
    prices = _make_prices(10)

    # Day 0: allocate equally.  Days 1-9: STAY everything.
    rows = []
    for i in range(len(prices)):
        if i == 0:
            rows.append({"AAPL": 0.33, "MSFT": 0.33, "GOOGL": 0.34})
        else:
            rows.append({"AAPL": STAY, "MSFT": STAY, "GOOGL": STAY})
    signals = pd.DataFrame(rows, index=prices.index)

    result = backtest(prices, signals, initial_cash=10_000, commission=0.0, slippage=0.0)

    # Only the first day should have trades (3 buys)
    assert len(result.trades) == 3, f"expected 3 trades, got {len(result.trades)}"

    # Shares must be constant after day 0
    pos = result.positions()
    for asset in ("AAPL", "MSFT", "GOOGL"):
        shares_series = pos[asset]
        assert shares_series.iloc[1:].nunique() == 1, \
            f"{asset} shares changed during STAY: {shares_series.tolist()}"

    print(f"  STAY test passed: {result}")


def test_partial_stay():
    """STAY some assets, rebalance others — STAY shares must not change."""
    prices = _make_prices(5)

    signals = pd.DataFrame({
        "AAPL":  [0.5,   STAY, STAY, 0.3,  STAY],
        "MSFT":  [0.3,   STAY, 0.8,  0.4,  STAY],
        "GOOGL": [0.2,   STAY, 0.2,  0.3,  STAY],
    }, index=prices.index)

    result = backtest(prices, signals, initial_cash=10_000, commission=0.0, slippage=0.0)

    pos = result.positions()
    # Between day 0 and day 1 (all STAY): no trades
    for a in ("AAPL", "MSFT", "GOOGL"):
        assert pos[a].iloc[0] == pos[a].iloc[1], f"{a} moved on all-STAY day"

    # Day 2: AAPL is STAY, MSFT/GOOGL rebalance
    assert pos["AAPL"].iloc[1] == pos["AAPL"].iloc[2], \
        "AAPL should not trade when marked STAY on day 2"

    print(f"  partial STAY test passed: {result}")


def test_zero_signal_closes_position():
    prices = _make_prices(5)

    signals = pd.DataFrame({
        "AAPL":  [0.5, 0.0,  0.0,  0.0,  0.0],
        "MSFT":  [0.3, 1.0,  1.0,  1.0,  1.0],
        "GOOGL": [0.2, 0.0,  0.0,  0.0,  0.0],
    }, index=prices.index)

    result = backtest(prices, signals, initial_cash=10_000, commission=0.0, slippage=0.0)
    pos = result.positions()

    # After day 1: AAPL and GOOGL should be ~0 shares
    assert abs(pos["AAPL"].iloc[-1]) < 1e-6
    assert abs(pos["GOOGL"].iloc[-1]) < 1e-6
    assert pos["MSFT"].iloc[-1] > 0

    print(f"  zero-signal close test passed: {result}")


def test_ohlcv_input():
    prices = _make_prices(10)
    highs = prices + 2
    lows = prices - 2

    data = OHLCV(close=prices, high=highs, low=lows)

    signals = pd.DataFrame({
        "AAPL":  [0.5] + [STAY] * 9,
        "MSFT":  [0.3] + [STAY] * 9,
        "GOOGL": [0.2] + [STAY] * 9,
    }, index=prices.index)

    result = backtest(data, signals, initial_cash=10_000)
    assert result.portfolio_value.iloc[0] > 0
    print(f"  OHLCV input test passed: {result}")


def test_limit_order():
    prices = _make_prices(10)
    low_price = prices["AAPL"].min()

    signals = pd.DataFrame({
        "AAPL":  [0.5] + [STAY] * 9,
        "MSFT":  [0.3] + [STAY] * 9,
        "GOOGL": [0.2] + [STAY] * 9,
    }, index=prices.index)

    oq = OrderQueue()
    oq.add(Order(
        asset="AAPL",
        side="buy",
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=low_price + 0.5,
        submit_date=prices.index[0],
    ))

    highs = prices + 2
    lows = prices - 2
    data = OHLCV(close=prices, high=highs, low=lows)

    result = backtest(data, signals, orders=oq, initial_cash=10_000, commission=0.0)
    assert len(result.trades) >= 2  # at least the MSFT+GOOGL market orders
    print(f"  limit order test passed: {result}")


def test_commission_models():
    prices = _make_prices(5)
    signals = pd.DataFrame({
        "AAPL":  [0.5] * 5,
        "MSFT":  [0.3] * 5,
        "GOOGL": [0.2] * 5,
    }, index=prices.index)

    r1 = backtest(prices, signals, commission=0.001)
    r2 = backtest(prices, signals, commission=IBKRTiered())
    r3 = backtest(prices, signals, commission=0.0)

    assert r3.metrics["total_commissions"] == 0.0
    assert r1.metrics["total_commissions"] > 0
    assert r2.metrics["total_commissions"] > 0
    print(f"  commission models test passed")


if __name__ == "__main__":
    test_basic_equal_weight()
    test_stay_freezes_shares_not_weights()
    test_partial_stay()
    test_zero_signal_closes_position()
    test_ohlcv_input()
    test_limit_order()
    test_commission_models()
    print("\nAll tests passed.")
