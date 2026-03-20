"""full-scale e2e: BM01 across all 30 buckets x 5 engines vs paper."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT / ".results"
VALIDATION_DIR = RESULTS_ROOT / "_e2e_validation" / "full"
BACKTEST_START = pd.Timestamp("2020-01-01")


def _month_starts(index):
    firsts, seen = set(), set()
    for d in index:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            firsts.add(d)
    return firsts


def _equal_weight(close_df, rebal_dates):
    ts = close_df.columns.tolist()
    w = 1.0 / len(ts)
    return {d: {t: w for t in ts} for d in rebal_dates if d in close_df.index}


def test_bm01_all_30_buckets(test_data):
    """BM01 equal-weight: all 30 buckets x 5 engines vs paper."""
    from crossengine.concordance import concordance

    buckets = test_data["buckets"]
    close_all = test_data["close"]
    close_all = close_all.loc[close_all.index >= BACKTEST_START]

    failures = []
    for bucket in buckets:
        bucket_id = bucket["bucket_id"]
        tickers = bucket["tickers"][:6]
        close = close_all[tickers]

        paper_dir = RESULTS_ROOT / "01-equal-weight" / "buckets" / bucket_id
        if not (paper_dir / "equity.csv").exists():
            continue

        report = concordance(
            _equal_weight, close, rebal_dates=_month_starts(close.index),
            initial_cash=100_000, commission=0.0015, slippage=0.0003,
        )
        paper = pd.read_csv(paper_dir / "equity.csv", index_col=0, parse_dates=True)

        for engine in report.engines:
            if engine not in paper.columns:
                continue
            common = paper[engine].index.intersection(report.equity[engine].index)
            max_abs = float(np.max(np.abs(paper[engine].loc[common].values - report.equity[engine].loc[common].values)))
            if max_abs > 0.01:
                failures.append(f"{bucket_id}/{engine}: ${max_abs:.6f}")

        ok = sum(1 for e in report.engines if e in paper.columns)
        print(f"  {bucket_id}: {ok}/{ok} engines exact match")

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    assert len(failures) == 0, f"{len(failures)} diverged: {failures[:5]}"
