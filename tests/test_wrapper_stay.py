"""tests for STAY support in engine wrappers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.types import STAY, SignalSchedule


def _make_prices(n_days: int = 20) -> pd.DataFrame:
    """deterministic price data."""
    np.random.seed(42)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    return pd.DataFrame({
        "A": 100 + np.cumsum(np.random.randn(n_days) * 1.0),
        "B": 200 + np.cumsum(np.random.randn(n_days) * 1.5),
    }, index=dates)


def test_ours_wrapper_stay_maps_to_engine_sentinel():
    """our wrapper maps STAY -> engine's 's' sentinel correctly."""
    from utils.wrappers.ours import run_ours

    close = _make_prices(10)
    ss: SignalSchedule = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[5]: {"A": STAY, "B": 0.7},
    }
    result = run_ours(close, ss, initial_cash=100_000, commission=0.001, slippage=0.0)

    # should produce a valid portfolio value series
    assert len(result.portfolio_value) == 10
    assert result.portfolio_value.iloc[0] > 0

    # STAY asset A should have constant shares between day 0 and day 5
    pos = result.positions()
    shares_a_day1 = pos["A"].iloc[1]
    shares_a_day4 = pos["A"].iloc[4]
    assert shares_a_day1 == shares_a_day4, "A shares changed during implicit STAY"

    # after day 5 rebalance, A shares should still be same (explicit STAY)
    shares_a_day5 = pos["A"].iloc[5]
    assert shares_a_day5 == shares_a_day4, "A shares changed on explicit STAY day"


def test_ours_wrapper_all_float_schedule():
    """our wrapper works with pure-float schedule (no STAY, backward compat)."""
    from utils.wrappers.ours import run_ours

    close = _make_prices(10)
    ss: SignalSchedule = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[5]: {"A": 0.3, "B": 0.7},
    }
    result = run_ours(close, ss, initial_cash=100_000, commission=0.001, slippage=0.0)
    assert len(result.portfolio_value) == 10
    assert result.portfolio_value.iloc[0] > 0


def test_ours_wrapper_partial_stay_no_trades():
    """STAY asset generates zero trades on rebalance day."""
    from utils.wrappers.ours import run_ours

    close = _make_prices(10)
    ss: SignalSchedule = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[5]: {"A": STAY, "B": 0.7},
    }
    result = run_ours(close, ss, initial_cash=100_000, commission=0.0, slippage=0.0)

    # filter trades on day 5
    day5_trades = result.trades[result.trades["date"] == close.index[5]]
    traded_assets = set(day5_trades["asset"].tolist())
    assert "A" not in traded_assets, f"A was traded on STAY day: {day5_trades}"


if __name__ == "__main__":
    test_ours_wrapper_stay_maps_to_engine_sentinel()
    test_ours_wrapper_all_float_schedule()
    test_ours_wrapper_partial_stay_no_trades()
    print("all wrapper STAY tests passed.")
