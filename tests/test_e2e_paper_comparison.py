"""e2e: concordance API vs paper results for BM01 bucket-01."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
PAPER_RESULTS = PROJECT / ".results" / "01-equal-weight" / "buckets" / "bucket-01"
VALIDATION_DIR = PROJECT / ".results" / "_e2e_validation"
BACKTEST_START = pd.Timestamp("2020-01-01")


def _month_starts(index):
    firsts, seen = set(), set()
    for d in index:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            firsts.add(d)
    return firsts


def test_concordance_api_matches_paper(test_data):
    """concordance API produces identical equity to paper for our engine."""
    from crossengine.concordance import concordance

    buckets = test_data["buckets"]
    close_all = test_data["close"]
    tickers = buckets[0]["tickers"]
    close = close_all.loc[close_all.index >= BACKTEST_START, tickers]
    rebal_dates = _month_starts(close.index)

    def equal_weight(close_df, rebal_dates_set):
        ts = close_df.columns.tolist()
        w = 1.0 / len(ts)
        return {d: {t: w for t in ts} for d in rebal_dates_set if d in close_df.index}

    report = concordance(
        equal_weight, close,
        rebal_dates=rebal_dates,
        initial_cash=100_000, commission=0.0015, slippage=0.0003,
        engines=("ours",),
    )

    if not (PAPER_RESULTS / "equity.csv").exists():
        import pytest
        pytest.skip("paper results not available")

    paper_equity = pd.read_csv(PAPER_RESULTS / "equity.csv", index_col=0, parse_dates=True)
    common = paper_equity["ours"].index.intersection(report.equity["ours"].index)
    max_abs_diff = float(np.max(np.abs(paper_equity["ours"].loc[common].values - report.equity["ours"].loc[common].values)))

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    report.to_json(VALIDATION_DIR / "bm01_bucket01_concordance.json")

    assert max_abs_diff < 0.01, f"equity differs by ${max_abs_diff:.6f}"
