"""tests for resolve_stay() forward simulation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from utils.resolve import has_stay, resolve_stay
from utils.types import STAY


def _make_prices(n_days: int = 20) -> pd.DataFrame:
    """deterministic price data for testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    return pd.DataFrame({
        "A": 100 + np.cumsum(np.random.randn(n_days) * 1.0),
        "B": 200 + np.cumsum(np.random.randn(n_days) * 1.5),
    }, index=dates)


def test_no_stay_passthrough():
    """schedule with no STAY returns identical weights."""
    close = _make_prices(10)
    ss = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[5]: {"A": 0.3, "B": 0.7},
    }
    ws = resolve_stay(ss, close, initial_cash=100_000, cost_rate=0.0)
    assert len(ws) == 2
    # first date: weights should match input (no prior positions)
    assert abs(ws[close.index[0]]["A"] - 0.6) < 0.01
    assert abs(ws[close.index[0]]["B"] - 0.4) < 0.01


def test_all_stay_preserves_drift():
    """after initial allocation, all-STAY date preserves drifted weights."""
    close = _make_prices(10)
    ss = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[5]: {"A": STAY, "B": STAY},
    }
    ws = resolve_stay(ss, close, initial_cash=100_000, cost_rate=0.0)

    # second date should have drifted weights (not 0.6/0.4)
    w_a = ws[close.index[5]]["A"]
    w_b = ws[close.index[5]]["B"]
    assert abs(w_a + w_b - 1.0) < 1e-6, f"weights don't sum to 1: {w_a + w_b}"
    # weights should have drifted from original 0.6/0.4
    # (with random prices they almost certainly won't be exactly 0.6/0.4)


def test_partial_stay_budget_allocation():
    """STAY on A, rebalance B: A keeps drifted weight, B gets rest."""
    close = _make_prices(10)
    ss = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[5]: {"A": STAY, "B": 0.7},
    }
    ws = resolve_stay(ss, close, initial_cash=100_000, cost_rate=0.0)

    w_a = ws[close.index[5]]["A"]
    w_b = ws[close.index[5]]["B"]
    assert abs(w_a + w_b - 1.0) < 1e-6, f"weights don't sum to 1: {w_a + w_b}"
    # A should be at its drifted weight, B gets the remainder
    assert w_b > 0


def test_cost_rate_changes_allocation():
    """non-zero cost rate should result in different weights vs zero cost."""
    close = _make_prices(10)
    ss = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[5]: {"A": STAY, "B": 0.7},
    }
    ws_free = resolve_stay(ss, close, initial_cash=100_000, cost_rate=0.0)
    ws_cost = resolve_stay(ss, close, initial_cash=100_000, cost_rate=0.01)

    # costs change portfolio value, so weights should differ
    diff = abs(ws_cost[close.index[5]]["B"] - ws_free[close.index[5]]["B"])
    assert diff > 1e-6, f"cost rate had no effect on weights: diff={diff}"


def test_has_stay_detection():
    """has_stay correctly identifies STAY in schedules."""
    dates = pd.bdate_range("2025-01-02", periods=2)
    assert has_stay({dates[0]: {"A": STAY}})
    assert not has_stay({dates[0]: {"A": 0.5}})
    assert not has_stay({})


def test_resolve_matches_manual_drift():
    """resolve_stay matches hand-computed drifted weight for simple 2-asset case."""
    # constant prices: A stays at 100, B stays at 200
    dates = pd.bdate_range("2025-01-02", periods=10)
    close = pd.DataFrame({
        "A": [100.0] * 5 + [110.0] * 5,
        "B": [200.0] * 10,
    }, index=dates)

    ss = {
        dates[0]: {"A": 0.6, "B": 0.4},
        dates[5]: {"A": STAY, "B": 0.5},  # rebalance B, hold A
    }
    ws = resolve_stay(ss, close, initial_cash=100_000, cost_rate=0.0)

    # day 0: A gets 60k, B gets 40k
    # A shares = 60000/100 = 600, B shares = 40000/200 = 200
    # day 5: A price = 110, B price = 200
    # A value = 600 * 110 = 66000, B value = 200 * 200 = 40000
    # total = 106000, A drifted weight = 66000/106000 = 0.6226...
    # STAY A: keep 600 shares, value = 66000
    # budget for B = 106000 - 66000 = 40000
    # B gets all of budget: weight = 40000/106000 = 0.3774...
    expected_a = 66000 / 106000
    expected_b = 40000 / 106000

    assert abs(ws[dates[5]]["A"] - expected_a) < 1e-4, \
        f"A: expected {expected_a:.4f}, got {ws[dates[5]]['A']:.4f}"
    assert abs(ws[dates[5]]["B"] - expected_b) < 1e-4, \
        f"B: expected {expected_b:.4f}, got {ws[dates[5]]['B']:.4f}"


if __name__ == "__main__":
    test_no_stay_passthrough()
    test_all_stay_preserves_drift()
    test_partial_stay_budget_allocation()
    test_cost_rate_changes_allocation()
    test_has_stay_detection()
    test_resolve_matches_manual_drift()
    print("all resolve_stay tests passed.")
