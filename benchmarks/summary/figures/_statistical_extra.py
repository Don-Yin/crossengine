"""permutation test and q-q normality plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from summary.figures._common import (
    CATEGORY_COLORS,
    CATEGORY_ORDER,
    CATEGORY_TITLES,
    group_by_category,
    pub_style,
    save,
)
from summary.figures._statistical_helpers import (
    load_bucket_divergences,
    permutation_pvalue,
)

_SUBDIR = "statistical"


def plot_permutation_test(tables_stat_dir=None) -> None:
    """histogram of permuted test statistics with observed value per category."""
    bdf = load_bucket_divergences()
    if bdf is None:
        return
    groups = group_by_category(sorted(bdf["benchmark_id"].unique()))
    cats = [c for c in CATEGORY_ORDER if c in groups]
    if not cats:
        return
    _save_permutation_csv(bdf, groups, cats, tables_stat_dir)
    ncols, nrows = 2, -(-len(cats) // 2)
    with pub_style():
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows),
            layout="constrained", squeeze=False,
        )
        for ci, cat in enumerate(cats):
            r, c = divmod(ci, ncols)
            vals = bdf[bdf["benchmark_id"].isin(groups[cat])][
                "divergence"].dropna().values
            _draw_perm_panel(axes[r, c], vals, cat)
        for ci in range(len(cats), nrows * ncols):
            r, c = divmod(ci, ncols)
            axes[r, c].set_visible(False)
        fig.suptitle("permutation test (10,000 sign-flips)", fontsize=9)
        save(fig, "permutation-test.png", tight=True, subdir=_SUBDIR)


def _save_permutation_csv(bdf, groups, cats, tables_stat_dir) -> None:
    """persist permutation test results to csv."""
    import pandas as pd
    perm_records = []
    for cat in cats:
        vals = bdf[bdf["benchmark_id"].isin(groups[cat])]["divergence"].dropna().values
        if len(vals) >= 3:
            observed, _, p = permutation_pvalue(vals)
            perm_records.append({"category": cat, "observed_abs_mean": round(observed, 6),
                                 "p_value": round(p, 6), "n_obs": len(vals)})
    if perm_records and tables_stat_dir is not None:
        tables_stat_dir.mkdir(parents=True, exist_ok=True)
        perm_df = pd.DataFrame(perm_records)
        p_path = tables_stat_dir / "permutation-results.csv"
        perm_df.to_csv(p_path, index=False)
        print(f"  table -> {p_path}")


def _draw_perm_panel(ax, vals, cat) -> None:
    """draw one permutation histogram panel."""
    if len(vals) < 3:
        ax.set_title(CATEGORY_TITLES.get(cat, cat), fontsize=7)
        return
    observed, perm_dist, p = permutation_pvalue(vals)
    ax.hist(perm_dist, bins=50, color=CATEGORY_COLORS.get(cat, "#999"),
            alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(observed, color="#E64B35", linewidth=1.2, linestyle="--")
    ax.set_title(f"{CATEGORY_TITLES.get(cat, cat)} (p={p:.4f})", fontsize=7)
    ax.set_xlabel("|mean divergence|", fontsize=6)
    ax.tick_params(labelsize=5)


def plot_qq_normality() -> None:
    """q-q normality plots with shapiro-wilk test per category."""
    bdf = load_bucket_divergences()
    if bdf is None:
        return
    groups = group_by_category(sorted(bdf["benchmark_id"].unique()))
    cats = [c for c in CATEGORY_ORDER if c in groups]
    if not cats:
        return
    ncols, nrows = 2, -(-len(cats) // 2)
    with pub_style():
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows),
            layout="constrained", squeeze=False,
        )
        for ci, cat in enumerate(cats):
            r, c = divmod(ci, ncols)
            vals = bdf[bdf["benchmark_id"].isin(groups[cat])][
                "divergence"].dropna().values
            _draw_qq_panel(axes[r, c], vals, cat)
        for ci in range(len(cats), nrows * ncols):
            r, c = divmod(ci, ncols)
            axes[r, c].set_visible(False)
        fig.suptitle("q-q normality check (shapiro-wilk)", fontsize=9)
        save(fig, "qq-normality.png", tight=True, subdir=_SUBDIR)


def _draw_qq_panel(ax, vals, cat) -> None:
    """draw one q-q panel with shapiro-wilk annotation."""
    if len(vals) < 3:
        ax.set_title(CATEGORY_TITLES.get(cat, cat), fontsize=7)
        return
    (osm, osr), (slope, intercept, _) = stats.probplot(vals, dist="norm")
    ax.scatter(osm, osr, s=6, color=CATEGORY_COLORS.get(cat, "#999"),
               alpha=0.6, edgecolors="none")
    ax.plot(osm, slope * np.array(osm) + intercept, "-",
            color="#333", linewidth=0.8)
    sample = vals[:5000] if len(vals) > 5000 else vals
    sw_stat, sw_p = stats.shapiro(sample)
    ax.set_title(
        f"{CATEGORY_TITLES.get(cat, cat)}\nW={sw_stat:.4f}, p={sw_p:.4f}",
        fontsize=6,
    )
    ax.set_xlabel("theoretical quantiles", fontsize=5)
    ax.set_ylabel("sample quantiles", fontsize=5)
    ax.tick_params(labelsize=5)
