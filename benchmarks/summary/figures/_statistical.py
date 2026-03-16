"""statistical robustness figures for the engine comparison study."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy import stats

from summary.collect import SUMMARY_DIR
from summary.figures._common import (
    CATEGORY_COLORS,
    CATEGORY_ORDER,
    CATEGORY_TITLES,
    category,
    group_by_category,
    pub_style,
    save,
    short_label,
)
from summary.figures._statistical_helpers import (
    load_bucket_divergences,
    tost_power,
    tost_pvalue,
)

_TABLES_STAT_DIR = SUMMARY_DIR / "tables" / "raw" / "statistical"

_SUBDIR = "statistical"


def plot_tost_equivalence() -> None:
    """heatmap of tost equivalence at 10 bps and 50 bps margins."""
    bdf = load_bucket_divergences()
    if bdf is None:
        return
    pairs = sorted(bdf["pair"].unique())
    bids = sorted(bdf["benchmark_id"].unique())
    grid = np.full((len(bids), len(pairs)), 3, dtype=int)
    pvals = np.full((len(bids), len(pairs)), np.nan)
    bid_idx = {b: i for i, b in enumerate(bids)}
    pair_idx = {p: j for j, p in enumerate(pairs)}
    for (bid, pair), grp in bdf.groupby(["benchmark_id", "pair"]):
        _fill_tost_cell(grid, pvals, bid_idx[bid], pair_idx[pair], grp["divergence"].dropna().values)
    _save_tost_csv(bids, pairs, grid, pvals)
    labels = [short_label(b) for b in bids]
    plabels = [p.replace("_vs_", " vs ") for p in pairs]
    cmap = ListedColormap(["#FADBD8", "#E8967A", "#E64B35", "#D9D9D9"])
    with pub_style():
        fig, ax = plt.subplots(
            figsize=(0.9 * len(pairs) + 2, 0.3 * len(bids) + 1.2),
            layout="constrained",
        )
        ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=3)
        _annotate_pvals(ax, pvals)
        ax.set_yticks(range(len(bids)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(plabels, fontsize=6, rotation=35, ha="right")
        ax.set_title(
            "tost equivalence (light=10bps, mid=50bps, dark=not equiv., grey=no data)",
            fontsize=7,
        )
        save(fig, "tost-equivalence.png", tight=True, subdir=_SUBDIR)


def _fill_tost_cell(grid, pvals, i, j, vals) -> None:
    """evaluate tost at both margins for one benchmark-pair cell."""
    if len(vals) < 3:
        return
    p10, p50 = tost_pvalue(vals, 0.10), tost_pvalue(vals, 0.50)
    grid[i, j] = 0 if p10 < 0.05 else (1 if p50 < 0.05 else 2)
    pvals[i, j] = min(p10, p50)


def _save_tost_csv(bids, pairs, grid, pvals) -> None:
    """persist tost results to csv for latex tables."""
    _TABLES_STAT_DIR.mkdir(parents=True, exist_ok=True)
    status_map = {0: "equiv_10bps", 1: "equiv_50bps", 2: "not_equiv", 3: "no_data"}
    rows = []
    for i, bid in enumerate(bids):
        for j, pair in enumerate(pairs):
            rows.append(
                {
                    "benchmark_id": bid,
                    "pair": pair,
                    "status": status_map.get(grid[i, j], "unknown"),
                    "p_value": round(float(pvals[i, j]), 6) if np.isfinite(pvals[i, j]) else None,
                }
            )
    df = pd.DataFrame(rows)
    p = _TABLES_STAT_DIR / "tost-results.csv"
    df.to_csv(p, index=False)
    print(f"  table -> {p}")


def _annotate_pvals(ax, pvals) -> None:
    """write p-values into heatmap cells."""
    for i, j in np.ndindex(pvals.shape):
        v = pvals[i, j]
        txt = f"{v:.3f}" if np.isfinite(v) else "n/a"
        ax.text(j, i, txt, ha="center", va="center", fontsize=5, color="white", fontweight="bold")


def plot_power_analysis() -> None:
    """line chart of tost power across equivalence margins per category."""
    bdf = load_bucket_divergences()
    if bdf is None:
        return
    margins_bps = [5, 10, 25, 50, 100, 150, 200]
    margins_pct = [m / 100 for m in margins_bps]
    groups = group_by_category(sorted(bdf["benchmark_id"].unique()))
    cat_stds = _compute_cat_stds(bdf, groups)
    with pub_style():
        fig, ax = plt.subplots(figsize=(3.5, 2.8), layout="constrained")
        for cat in CATEGORY_ORDER:
            if cat not in cat_stds:
                continue
            powers = [tost_power(30, cat_stds[cat], m) for m in margins_pct]
            ax.plot(margins_bps, powers, "o-", color=CATEGORY_COLORS.get(cat, "#333"), label=CATEGORY_TITLES.get(cat, cat), markersize=4, linewidth=1.2)
        ax.axhline(0.8, ls="--", color="#555", linewidth=0.8, label="80% power")
        ax.set_xlabel("equivalence margin (bps)")
        ax.set_ylabel("statistical power")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=6, loc="lower right")
        ax.set_title("tost power analysis (n = 30 buckets)", fontsize=9)
        save(fig, "power-analysis.png", tight=True, subdir=_SUBDIR)


def _compute_cat_stds(bdf, groups) -> dict[str, float]:
    """compute per-category divergence standard deviation."""
    result = {}
    for cat, bids in groups.items():
        vals = bdf[bdf["benchmark_id"].isin(bids)]["divergence"].dropna().values
        if len(vals) > 1:
            result[cat] = float(np.std(vals, ddof=1))
    return result


def plot_wilcoxon_robustness() -> None:
    """scatter comparing t-test vs wilcoxon signed-rank p-values."""
    bdf = load_bucket_divergences()
    if bdf is None:
        return
    records: list[dict] = []
    for (bid, pair), grp in bdf.groupby(["benchmark_id", "pair"]):
        _append_wilcoxon_record(records, bid, pair, grp["divergence"].dropna().values)
    if not records:
        return
    rdf = pd.DataFrame(records)
    _TABLES_STAT_DIR.mkdir(parents=True, exist_ok=True)
    p = _TABLES_STAT_DIR / "wilcoxon-results.csv"
    rdf.to_csv(p, index=False)
    print(f"  table -> {p}")
    with pub_style():
        fig, ax = plt.subplots(figsize=(4.5, 4.5), layout="constrained")
        for cat in CATEGORY_ORDER:
            csub = rdf[rdf["category"] == cat]
            if csub.empty:
                continue
            ax.scatter(csub["t_p"], csub["w_p"], s=14, alpha=0.7, color=CATEGORY_COLORS.get(cat, "#333"), label=CATEGORY_TITLES.get(cat, cat), edgecolors="none")
        ax.plot([0, 1], [0, 1], "--", color="#aaa", linewidth=0.8)
        ax.set_xlabel("t-test p-value")
        ax.set_ylabel("wilcoxon p-value")
        ax.set_title("parametric vs non-parametric agreement", fontsize=9)
        ax.legend(fontsize=5.5, loc="lower right")
        save(fig, "wilcoxon-robustness.png", tight=True, subdir=_SUBDIR)


def _append_wilcoxon_record(records, bid, pair, vals) -> None:
    """compute t-test and wilcoxon p-values for one benchmark-pair."""
    if len(vals) < 3:
        return
    t_p = float(stats.ttest_1samp(vals, 0).pvalue)
    nonzero = vals[vals != 0]
    if len(nonzero) >= 6:
        w_p = float(stats.wilcoxon(vals, zero_method="wilcox", correction=True).pvalue)
    else:
        w_p = 1.0
    records.append({"benchmark_id": bid, "pair": pair, "t_p": t_p, "w_p": w_p, "category": category(bid)})


def plot_permutation_test() -> None:
    """delegated to _statistical_extra for LOC compliance."""
    from summary.figures._statistical_extra import plot_permutation_test as _impl

    _impl(tables_stat_dir=_TABLES_STAT_DIR)


def plot_qq_normality() -> None:
    """delegated to _statistical_extra for LOC compliance."""
    from summary.figures._statistical_extra import plot_qq_normality as _impl

    _impl()
