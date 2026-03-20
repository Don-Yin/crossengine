"""e2e: 5-engine concordance API vs paper results for BM01 and BM02."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT / ".results"
VALIDATION_DIR = RESULTS_ROOT / "_e2e_validation"
BACKTEST_START = pd.Timestamp("2020-01-01")


def _month_starts(index):
    firsts, seen = set(), set()
    for d in index:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            firsts.add(d)
    return firsts


def _compare(paper_s, api_s):
    common = paper_s.index.intersection(api_s.index)
    pv, av = paper_s.loc[common].values, api_s.loc[common].values
    return {
        "max_abs_diff": float(np.max(np.abs(pv - av))),
        "correlation": float(np.corrcoef(pv, av)[0, 1]) if len(pv) > 1 else 1.0,
    }


def test_bm01_5engine(test_data):
    """BM01 equal-weight: all engines vs paper."""
    from crossengine.concordance import concordance

    buckets = test_data["buckets"]
    close_all = test_data["close"]
    tickers = buckets[0]["tickers"]
    close = close_all.loc[close_all.index >= BACKTEST_START, tickers]

    def equal_weight(c, dates):
        ts = c.columns.tolist()
        w = 1.0 / len(ts)
        return {d: {t: w for t in ts} for d in dates if d in c.index}

    report = concordance(equal_weight, close, rebal_dates=_month_starts(close.index),
                         initial_cash=100_000, commission=0.0015, slippage=0.0003)

    paper_path = RESULTS_ROOT / "01-equal-weight" / "buckets" / "bucket-01" / "equity.csv"
    if not paper_path.exists():
        pytest.skip("paper results not available")

    paper = pd.read_csv(paper_path, index_col=0, parse_dates=True)
    for engine in report.engines:
        if engine in paper.columns:
            r = _compare(paper[engine], report.equity[engine])
            assert r["correlation"] > 0.9999, f"{engine}: corr {r['correlation']}"


def test_bm02_5engine(test_data):
    """BM02 stay-drift: all engines vs paper."""
    from crossengine.concordance import STAY, concordance

    buckets = test_data["buckets"]
    close_all = test_data["close"]
    tickers = buckets[0]["tickers"][:2]
    close = close_all.loc[close_all.index >= BACKTEST_START, tickers]

    ss = {
        close.index[0]: {tickers[0]: 0.6, tickers[1]: 0.4},
        close.index[60]: {tickers[0]: STAY, tickers[1]: 0.7},
    }
    report = concordance(ss, close, initial_cash=100_000, commission=0.0015, slippage=0.0003)

    paper_path = RESULTS_ROOT / "02-stay-drift" / "buckets" / "bucket-01" / "equity.csv"
    if not paper_path.exists():
        pytest.skip("paper results not available")

    paper = pd.read_csv(paper_path, index_col=0, parse_dates=True)
    for engine in report.engines:
        if engine in paper.columns:
            r = _compare(paper[engine], report.equity[engine])
            assert r["correlation"] > 0.9999, f"{engine}: corr {r['correlation']}"
