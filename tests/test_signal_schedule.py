"""tests for SignalSchedule type and STAY sentinel."""
import sys
from pathlib import Path

import pandas as pd

# make benchmarks/utils importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from utils.types import STAY, SignalSchedule, WeightSchedule


def test_stay_sentinel_value():
    """STAY sentinel is the string 'STAY'."""
    assert STAY == "STAY"
    assert isinstance(STAY, str)


def test_signal_schedule_accepts_stay():
    """SignalSchedule values can be float or STAY."""
    dates = pd.bdate_range("2025-01-02", periods=3)
    ss: SignalSchedule = {
        dates[0]: {"AAPL": 0.6, "MSFT": 0.4},
        dates[1]: {"AAPL": STAY, "MSFT": 0.7},
        dates[2]: {"AAPL": STAY, "MSFT": STAY},
    }
    assert ss[dates[1]]["AAPL"] == STAY
    assert ss[dates[0]]["AAPL"] == 0.6


def test_weight_schedule_is_pure_float():
    """WeightSchedule only holds floats (no STAY)."""
    dates = pd.bdate_range("2025-01-02", periods=2)
    ws: WeightSchedule = {
        dates[0]: {"AAPL": 0.6, "MSFT": 0.4},
        dates[1]: {"AAPL": 0.3, "MSFT": 0.7},
    }
    for row in ws.values():
        for v in row.values():
            assert isinstance(v, float)


def test_pure_float_schedule_is_valid_signal_schedule():
    """a WeightSchedule (all floats) is a valid SignalSchedule."""
    dates = pd.bdate_range("2025-01-02", periods=2)
    ws: WeightSchedule = {
        dates[0]: {"AAPL": 0.5, "MSFT": 0.5},
        dates[1]: {"AAPL": 0.3, "MSFT": 0.7},
    }
    # should be usable wherever SignalSchedule is expected
    ss: SignalSchedule = ws
    assert len(ss) == 2


def test_has_stay_detection():
    """utility pattern: detect whether a schedule contains STAY."""
    dates = pd.bdate_range("2025-01-02", periods=2)
    ss_with_stay: SignalSchedule = {
        dates[0]: {"AAPL": 0.6, "MSFT": 0.4},
        dates[1]: {"AAPL": STAY, "MSFT": 0.7},
    }
    ss_no_stay: SignalSchedule = {
        dates[0]: {"AAPL": 0.6, "MSFT": 0.4},
        dates[1]: {"AAPL": 0.3, "MSFT": 0.7},
    }
    has_stay = lambda ss: any(v == STAY for row in ss.values() for v in row.values())
    assert has_stay(ss_with_stay)
    assert not has_stay(ss_no_stay)


if __name__ == "__main__":
    test_stay_sentinel_value()
    test_signal_schedule_accepts_stay()
    test_weight_schedule_is_pure_float()
    test_pure_float_schedule_is_valid_signal_schedule()
    test_has_stay_detection()
    print("all SignalSchedule tests passed.")
