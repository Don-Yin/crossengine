"""tests for the concordance() public API and ConcordanceReport."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from crossengine.concordance import STAY, ConcordanceReport, SignalSchedule, concordance


def _make_prices(n_days: int = 60) -> pd.DataFrame:
    """deterministic price data."""
    np.random.seed(42)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    return pd.DataFrame({
        "A": 100 + np.cumsum(np.random.randn(n_days) * 1.0),
        "B": 200 + np.cumsum(np.random.randn(n_days) * 1.5),
    }, index=dates)


def test_concordance_with_dict():
    """concordance() accepts a pre-computed SignalSchedule dict."""
    close = _make_prices()
    ss: SignalSchedule = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[20]: {"A": 0.5, "B": 0.5},
    }
    report = concordance(ss, close, engines=("ours",))
    assert isinstance(report, ConcordanceReport)
    assert "ours" in report.engines
    assert len(report.equity) == len(close)


def test_concordance_with_callable():
    """concordance() accepts a strategy callable."""
    close = _make_prices()

    def equal_weight(close: pd.DataFrame, rebal_dates: set[pd.Timestamp]) -> SignalSchedule:
        tickers = close.columns.tolist()
        w = 1.0 / len(tickers)
        return {d: {t: w for t in tickers} for d in rebal_dates if d in close.index}

    report = concordance(equal_weight, close, engines=("ours",))
    assert isinstance(report, ConcordanceReport)
    assert len(report.engines) == 1


def test_concordance_with_stay():
    """concordance() handles STAY in SignalSchedule."""
    close = _make_prices()
    ss: SignalSchedule = {
        close.index[0]: {"A": 0.6, "B": 0.4},
        close.index[30]: {"A": STAY, "B": 0.7},
    }
    report = concordance(ss, close, engines=("ours",))
    assert isinstance(report, ConcordanceReport)
    assert report.equity["ours"].iloc[-1] > 0


def test_report_summary():
    """ConcordanceReport.summary() returns readable text."""
    close = _make_prices()
    ss = {close.index[0]: {"A": 0.5, "B": 0.5}}
    report = concordance(ss, close, engines=("ours",))
    text = report.summary()
    assert "engine concordance report" in text
    assert "ours" in text


def test_report_max_divergence_single_engine():
    """max_divergence is 0 with a single engine."""
    close = _make_prices()
    ss = {close.index[0]: {"A": 0.5, "B": 0.5}}
    report = concordance(ss, close, engines=("ours",))
    assert report.max_divergence == 0.0


def test_report_to_json():
    """ConcordanceReport.to_json() writes valid JSON."""
    close = _make_prices()
    ss = {close.index[0]: {"A": 0.5, "B": 0.5}}
    report = concordance(ss, close, engines=("ours",))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        report.to_json(f.name)
        data = json.loads(Path(f.name).read_text())
    assert "engines" in data
    assert "divergence" in data
    assert "max_divergence_pct" in data


def test_report_repr():
    """ConcordanceReport has informative repr."""
    close = _make_prices()
    ss = {close.index[0]: {"A": 0.5, "B": 0.5}}
    report = concordance(ss, close, engines=("ours",))
    r = repr(report)
    assert "ConcordanceReport" in r
    assert "ours" in r


def test_report_engine_sensitivity():
    """engine_sensitivity property returns a float."""
    close = _make_prices()
    ss = {close.index[0]: {"A": 0.5, "B": 0.5}}
    report = concordance(ss, close, engines=("ours",))
    assert isinstance(report.engine_sensitivity, float)


def test_concordance_custom_rebal_dates():
    """concordance() uses custom rebal_dates when provided."""
    close = _make_prices()
    custom_dates = {close.index[0], close.index[10], close.index[30]}

    def strategy(close, rebal_dates):
        tickers = close.columns.tolist()
        w = 1.0 / len(tickers)
        return {d: {t: w for t in tickers} for d in rebal_dates if d in close.index}

    report = concordance(strategy, close, rebal_dates=custom_dates, engines=("ours",))
    assert isinstance(report, ConcordanceReport)


def test_concordance_empty_schedule_raises():
    """concordance() raises ValueError for empty schedule."""
    close = _make_prices()
    try:
        concordance({}, close, engines=("ours",))
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()


if __name__ == "__main__":
    test_concordance_with_dict()
    test_concordance_with_callable()
    test_concordance_with_stay()
    test_report_summary()
    test_report_max_divergence_single_engine()
    test_report_to_json()
    test_report_repr()
    test_report_engine_sensitivity()
    test_concordance_custom_rebal_dates()
    test_concordance_empty_schedule_raises()
    print("all concordance API tests passed.")
