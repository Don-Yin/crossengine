"""one-sample t-test significance markers for distribution figures.

runs per-metric, per-benchmark t-tests against scientifically motivated
reference values, applies Benjamini-Hochberg FDR correction within each
figure family, and annotates cells with +/++/+++ markers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

# metric -> (reference value, scipy alternative param)
_CONFIG: dict[str, tuple[float, str]] = {
    "cagr_pct":          (0,  "greater"),
    "sharpe_ratio":      (0,  "greater"),
    "sortino_ratio":     (0,  "greater"),
    "calmar_ratio":      (0,  "greater"),
    "win_rate_pct":      (50, "greater"),
    "excess_return_pct": (0,  "greater"),
    "alpha_ann_pct":     (0,  "greater"),
    "beta":              (1,  "two-sided"),
    "information_ratio": (0,  "greater"),
    "up_capture":        (1,  "two-sided"),
    "down_capture":      (1,  "less"),
}

_TIERS = [(0.001, "+++"), (0.01, "++"), (0.05, "+")]


def _to_marker(p: float) -> str:
    """convert corrected p-value to significance tier string."""
    for thresh, m in _TIERS:
        if p < thresh:
            return m
    return ""


def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction, returning adjusted p-values."""
    m = len(pvals)
    if m == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = np.empty(m)
    ranked[order] = np.arange(1, m + 1)
    adjusted = np.minimum(pvals * m / ranked, 1.0)
    adjusted[order[::-1]] = np.minimum.accumulate(adjusted[order[::-1]])
    return adjusted


def _run_test(vals: np.ndarray, ref: float, alt: str) -> float:
    """one-sample t-test p-value; returns 1.0 if insufficient data."""
    clean = vals[np.isfinite(vals)]
    if len(clean) < 3 or np.std(clean) < 1e-12:
        return 1.0
    return ttest_1samp(clean, popmean=ref, alternative=alt).pvalue


def _collect_pvalues(
    mdata: pd.DataFrame, cfg: tuple[float, str],
    groups: list[str], metric: str,
    keys: list[tuple], raw_p: list[float],
) -> None:
    """run t-tests for all groups of one metric, appending to keys/raw_p."""
    ref, alt = cfg
    for name, grp in mdata.groupby(groups):
        vals = grp["value"].dropna().values
        key = (metric,) + (name if isinstance(name, tuple) else (name,))
        keys.append(key)
        raw_p.append(_run_test(vals, ref, alt))


def compute_significance(
    long: pd.DataFrame, metrics: list[str], split: bool,
) -> dict[tuple, str]:
    """compute BH-FDR corrected significance markers for all testable cells."""
    keys: list[tuple] = []
    raw_p: list[float] = []
    groups = ["label", "section"] if split else ["label"]

    for metric in metrics:
        cfg = _CONFIG.get(metric)
        if cfg is None:
            continue
        mdata = long[long["metric"] == metric]
        _collect_pvalues(mdata, cfg, groups, metric, keys, raw_p)

    if not keys:
        return {}

    corrected = _bh_correct(np.array(raw_p))
    return {k: _to_marker(p) for k, p in zip(keys, corrected)}


def annotate_significance(
    ax, metric: str, labels: list[str],
    sig_map: dict[tuple, str], split: bool,
) -> None:
    """place significance markers at the right edge of each cell."""
    xlim = ax.get_xlim()
    x = xlim[1] - (xlim[1] - xlim[0]) * 0.015

    if split:
        _annotate_split(ax, metric, labels, sig_map, x)
    else:
        _annotate_single(ax, metric, labels, sig_map, x)


def _annotate_single(ax, metric: str, labels: list[str], sig_map: dict, x: float) -> None:
    """one marker per row for single-distribution plots."""
    for idx, lbl in enumerate(labels):
        txt = sig_map.get((metric, lbl), "")
        if txt:
            ax.text(x, idx, txt, fontsize=4.5, ha="right", va="center",
                    fontweight="bold", color="#2D7D46")


def _annotate_split(ax, metric: str, labels: list[str], sig_map: dict, x: float) -> None:
    """two markers per row for split-distribution plots (SPX above, asset_avg below)."""
    for idx, lbl in enumerate(labels):
        spx_m = sig_map.get((metric, lbl, "spx"), "")
        avg_m = sig_map.get((metric, lbl, "asset_avg"), "")
        if spx_m:
            ax.text(x, idx - 0.16, spx_m, fontsize=4, ha="right",
                    va="center", fontweight="bold", color="#4477AA")
        if avg_m:
            ax.text(x, idx + 0.16, avg_m, fontsize=4, ha="right",
                    va="center", fontweight="bold", color="#EE6677")
