"""tests for run_benchmark() orchestrator STAY routing."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.resolve import has_stay, resolve_stay
from utils.types import STAY, SignalSchedule, WeightSchedule


def _make_prices(n_days: int = 20) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    return pd.DataFrame({
        "A": 100 + np.cumsum(np.random.randn(n_days) * 1.0),
        "B": 200 + np.cumsum(np.random.randn(n_days) * 1.5),
    }, index=dates)


def test_routing_no_stay():
    """pure-float schedule: has_stay returns False, no resolution needed."""
    close = _make_prices(10)
    ss: SignalSchedule = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[5]: {"A": 0.3, "B": 0.7},
    }
    assert not has_stay(ss)
    # resolve_stay on a no-STAY schedule returns equivalent weights
    ws = resolve_stay(ss, close, initial_cash=100_000, cost_rate=0.0)
    assert len(ws) == 2
    assert abs(ws[close.index[0]]["A"] - 0.6) < 0.01


def test_routing_with_stay():
    """schedule with STAY: has_stay returns True, resolve produces pure floats."""
    close = _make_prices(10)
    ss: SignalSchedule = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[5]: {"A": STAY, "B": 0.7},
    }
    assert has_stay(ss)
    ws = resolve_stay(ss, close, initial_cash=100_000, cost_rate=0.0)
    # resolved schedule should have no STAY
    for row in ws.values():
        for v in row.values():
            assert isinstance(v, float), f"STAY not resolved: {v}"


def test_category_a_gets_raw_schedule():
    """category A engines (ours) receive the raw SignalSchedule with STAY."""
    from utils.wrappers.ours import run_ours

    close = _make_prices(10)
    ss: SignalSchedule = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[5]: {"A": STAY, "B": 0.7},
    }
    # ours wrapper accepts SignalSchedule with STAY directly
    result = run_ours(close, ss, initial_cash=100_000, commission=0.001, slippage=0.0)
    assert len(result.portfolio_value) == 10


def test_category_b_gets_resolved_schedule():
    """category B engines receive pre-resolved WeightSchedule (pure floats)."""
    close = _make_prices(10)
    ss: SignalSchedule = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[5]: {"A": STAY, "B": 0.7},
    }
    ws = resolve_stay(ss, close, initial_cash=100_000, cost_rate=0.0018)
    # vectorbt wrapper signature: (close, ws, *, initial_cash, commission)
    # just verify the resolved schedule is valid input format
    for d, row in ws.items():
        assert isinstance(d, pd.Timestamp)
        for t, v in row.items():
            assert isinstance(v, float), f"non-float in resolved schedule: {v}"
            assert isinstance(t, str)


def test_backward_compat_pure_float():
    """existing benchmarks pass pure-float dicts; should work unchanged."""
    close = _make_prices(10)
    ws: WeightSchedule = {
        close.index[0]: {"A": 0.5, "B": 0.5},
    }
    # WeightSchedule is valid as SignalSchedule
    ss: SignalSchedule = ws
    assert not has_stay(ss)
    from utils.wrappers.ours import run_ours
    result = run_ours(close, ss, initial_cash=100_000, commission=0.001, slippage=0.0)
    assert len(result.portfolio_value) == 10


if __name__ == "__main__":
    test_routing_no_stay()
    test_routing_with_stay()
    test_category_a_gets_raw_schedule()
    test_category_b_gets_resolved_schedule()
    test_backward_compat_pure_float()
    print("all orchestrator tests passed.")
