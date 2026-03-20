"""tests for simplified BM02 stay-drift using SignalSchedule."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.types import STAY, SignalSchedule
from utils.resolve import has_stay, resolve_stay


def _make_prices(n_days: int = 100) -> pd.DataFrame:
    """deterministic 2-asset price data."""
    np.random.seed(42)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    return pd.DataFrame({
        "A": 100 + np.cumsum(np.random.randn(n_days) * 1.0),
        "B": 200 + np.cumsum(np.random.randn(n_days) * 1.5),
    }, index=dates)


def test_bm02_signal_schedule():
    """BM02-style schedule with STAY is valid and detectable."""
    close = _make_prices()
    ss: SignalSchedule = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[60]: {"A": STAY, "B": 0.7},
    }
    assert has_stay(ss)
    assert len(ss) == 2


def test_bm02_resolve_produces_drifted_weight():
    """resolve_stay on BM02 schedule produces correct drifted weight for A."""
    close = _make_prices()
    ss: SignalSchedule = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[60]: {"A": STAY, "B": 0.7},
    }
    ws = resolve_stay(ss, close, initial_cash=100_000, cost_rate=0.0)

    # day 0: A=0.6, B=0.4
    assert abs(ws[close.index[0]]["A"] - 0.6) < 0.01

    # day 60: A drifted, B gets remainder
    w_a = ws[close.index[60]]["A"]
    w_b = ws[close.index[60]]["B"]
    assert abs(w_a + w_b - 1.0) < 1e-6
    assert w_a > 0  # A should have positive drifted weight
    assert w_b > 0  # B should get the rest


def test_bm02_ours_engine_stay():
    """our engine handles BM02-style STAY correctly."""
    from utils.wrappers.ours import run_ours

    close = _make_prices()
    ss: SignalSchedule = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[60]: {"A": STAY, "B": 0.7},
    }
    result = run_ours(close, ss, initial_cash=100_000, commission=0.0015, slippage=0.0003)
    assert len(result.portfolio_value) == len(close)

    # A shares should be constant between day 0 and day 60
    pos = result.positions()
    a_shares_day1 = pos["A"].iloc[1]
    a_shares_day59 = pos["A"].iloc[59]
    assert a_shares_day1 == a_shares_day59, "A shares changed before rebalance"

    # A shares should remain the same on day 60 (STAY)
    a_shares_day60 = pos["A"].iloc[60]
    assert a_shares_day60 == a_shares_day59, "A shares changed on STAY day"


def test_bm02_reduction_from_175_to_under_60_loc():
    """the new BM02 run.py should be under 60 lines."""
    bm02_path = Path(__file__).resolve().parent.parent / "benchmarks" / "02-stay-drift" / "run.py"
    lines = bm02_path.read_text().strip().split("\n")
    assert len(lines) <= 60, f"BM02 is {len(lines)} lines, expected 60 or fewer"


if __name__ == "__main__":
    test_bm02_signal_schedule()
    test_bm02_resolve_produces_drifted_weight()
    test_bm02_ours_engine_stay()
    test_bm02_reduction_from_175_to_under_60_loc()
    print("all BM02 simplified tests passed.")
