"""pseudo-replication and floor decomposition diagnostics."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from summary.collect import load_all_equity
from utils.wrappers import ACTIVE_ENGINES


def compute_bucket_autocorrelation(results_root=None) -> tuple:
    """compute lag-1 autocorrelation of bucket-level divergences for pseudo-replication assessment.

    for each (benchmark, pair), sorts 30 bucket divergences by bucket_id and
    computes lag-1 autocorrelation. near-zero values indicate exchangeability.
    note: bucket ordering is lexicographic (bucket-01..bucket-30), not structural.
    """
    from summary.figures._statistical_helpers import load_bucket_divergences
    from summary.figures._common import category as cat_fn
    bdf = load_bucket_divergences()
    if bdf is None:
        return pd.DataFrame(), pd.DataFrame()
    autocorrs = []
    for (bid, pair), grp in bdf.groupby(["benchmark_id", "pair"]):
        vals = grp.sort_values("bucket_id")["divergence"].dropna().values
        if len(vals) < 10:
            continue
        centered = vals - vals.mean()
        denom = np.sum(centered ** 2)
        ac1 = float(np.sum(centered[:-1] * centered[1:]) / denom) if denom > 1e-15 else 0.0
        autocorrs.append({
            "benchmark_id": bid, "pair": pair, "category": cat_fn(bid),
            "n_buckets": len(vals), "lag1_autocorr": round(ac1, 6),
        })
    detail_df = pd.DataFrame(autocorrs)
    cat_summary = _summarize_autocorr_by_category(detail_df)
    return detail_df, cat_summary


def _summarize_autocorr_by_category(detail_df: pd.DataFrame) -> pd.DataFrame:
    """summarize lag-1 autocorrelation by category."""
    records = []
    for cat, grp in detail_df.groupby("category"):
        acs = grp["lag1_autocorr"].values
        n = len(acs)
        if n < 2:
            continue
        mean_ac = float(np.mean(acs))
        se_ac = float(np.std(acs, ddof=1) / np.sqrt(n))
        lo = round(mean_ac - 1.96 * se_ac, 4)
        hi = round(mean_ac + 1.96 * se_ac, 4)
        records.append({
            "category": cat, "n_pairs": n,
            "mean_lag1_ac": round(mean_ac, 4),
            "se_lag1_ac": round(se_ac, 4),
            "ci95_lower": lo, "ci95_upper": hi,
            "interpretation": "independent" if lo <= 0 <= hi else "mild clustering",
        })
    return pd.DataFrame(records)


def compute_floor_decomposition(results_root=None) -> pd.DataFrame:
    """decompose pairwise divergence into equity-recording convention floor and cost-model residual.

    the floor is estimated per engine-pair from BM01 (simplest nonzero-cost strategy).
    pairs crossing the {backtrader, cvxportfolio} vs {ours, bt, vectorbt} boundary
    show a constant ~0.18% offset from equity-recording timing differences.
    """
    equities = load_all_equity(results_root) if results_root else load_all_equity()
    if not equities:
        return pd.DataFrame()
    engines = list(ACTIVE_ENGINES)
    baseline_bid = "01-equal-weight"
    baseline_divs = _compute_pair_divs(equities.get(baseline_bid), engines)
    records = []
    for bid, edf in sorted(equities.items()):
        pair_divs = _compute_pair_divs(edf, engines)
        for pair_key, raw_div in pair_divs.items():
            floor_raw = baseline_divs.get(pair_key, 0.0) if bid != baseline_bid else 0.0
            floor_est = min(floor_raw, raw_div)
            records.append({
                "benchmark_id": bid, "pair": pair_key,
                "raw_div_pct": round(raw_div, 6),
                "floor_pct": round(floor_est, 6),
                "cost_residual_pct": round(raw_div - floor_est, 6),
            })
    return pd.DataFrame(records)


def _compute_pair_divs(edf, engines) -> dict[str, float]:
    """compute total return divergence for all engine pairs from one benchmark."""
    if edf is None:
        return {}
    result = {}
    for ea, eb in itertools.combinations(engines, 2):
        if ea not in edf.columns or eb not in edf.columns:
            continue
        a, b = edf[ea].dropna(), edf[eb].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        ret_a = (a.iloc[-1] / a.iloc[0] - 1) * 100
        ret_b = (b.iloc[-1] / b.iloc[0] - 1) * 100
        result[f"{ea}_vs_{eb}"] = abs(ret_a - ret_b)
    return result
