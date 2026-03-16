"""implementation-risk analysis figures: divergence anatomy, rank concordance,
metric sensitivity, complexity analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from summary.concordance import compute_engine_concordance
from summary.figures._common import (
    CATEGORY_COLORS,
    add_category_legend,
    add_n_label,
    bid_color,
    category,
    get_n_buckets,
    pair_label,
    pub_style,
    save,
    short_label,
)
from summary.figures._mechanism import plot_economic_significance, plot_mechanism_comparison

__all__ = [
    "plot_divergence_anatomy",
    "plot_complexity_analysis",
    "plot_metric_sensitivity",
    "plot_mechanism_comparison",
    "plot_economic_significance",
]

_METRICS = ["total_return_pct", "cagr_pct", "ann_vol_pct", "sharpe", "max_dd_pct"]
_METRIC_LABELS = {
    "total_return_pct": "total return",
    "cagr_pct": "CAGR",
    "ann_vol_pct": "volatility",
    "sharpe": "Sharpe",
    "max_dd_pct": "max drawdown",
}


def _find_pair_col(cols, a, b):
    """find the divergence column matching a specific engine pair."""
    target = {a, b}
    for c in cols:
        parts = c.replace("div_", "").replace("_max_rel_pct", "").split("_vs_")
        if len(parts) == 2 and set(parts) == target:
            return c
    return None


def plot_divergence_anatomy(df: pd.DataFrame) -> None:
    """2-panel figure: quadrant scatter (bt vs vbt divergence) + driver fingerprint heatmap."""
    div_cols = sorted(c for c in df.columns if c.startswith("div_") and c.endswith("_max_rel_pct"))
    if len(div_cols) < 2:
        return

    sub = df.dropna(subset=div_cols).copy()
    sub["total_cost"] = sub.get("total_commissions", 0) + sub.get("total_slippage", 0)
    sub["per_trade_cost"] = sub["total_cost"] / sub["num_trades"].replace(0, np.nan)
    sub["is_ml"] = [1.0 if category(b) == "ml" else 0.0 for b in sub.index]

    with pub_style():
        fig, (ax_s, ax_h) = plt.subplots(1, 2, figsize=(10.5, 4.5), gridspec_kw={"width_ratios": [1.3, 1]}, layout="constrained")
        _draw_quadrant_scatter(ax_s, sub)
        _draw_driver_heatmap(ax_h, sub)
        save(fig, "divergence-anatomy.png", subdir="mechanism")


def _draw_quadrant_scatter(ax, sub) -> None:
    div_cols = sorted(c for c in sub.columns if c.startswith("div_") and c.endswith("_max_rel_pct"))
    col_bt = _find_pair_col(div_cols, "bt", "ours")
    col_vbt = _find_pair_col(div_cols, "ours", "vectorbt")
    if not col_bt or not col_vbt:
        return

    bt = sub[col_bt].values
    vbt = sub[col_vbt].values
    bt_log = np.log1p(bt * 1000)
    vbt_log = np.log1p(vbt * 1000)

    for bid, row in sub.iterrows():
        cat = category(bid)
        bx = np.log1p(row[col_bt] * 1000)
        vy = np.log1p(row[col_vbt] * 1000)
        sz = 25 + row.get("total_cost", 0) / 500
        ax.scatter(bx, vy, c=CATEGORY_COLORS.get(cat, "#333"), s=sz, edgecolors="white", linewidths=0.4, zorder=5, alpha=0.85)
        ax.annotate(short_label(bid), (bx, vy), fontsize=5, xytext=(3, 3), textcoords="offset points", color=bid_color(bid))

    lim = max(bt_log.max(), vbt_log.max()) * 1.15
    ax.plot([0, lim], [0, lim], "--", color="#AAA", linewidth=0.7)
    mid = np.log1p(0.03 * 1000)
    ax.axvspan(mid, lim, alpha=0.03, color="#E64B35", zorder=0)
    ax.axhspan(mid, lim, alpha=0.03, color="#3C5488", zorder=0)
    ax.text(0.88, 0.08, "bt-sensitive\n(cost-driven)", transform=ax.transAxes, fontsize=6, ha="center", color="#E64B35", alpha=0.7)
    ax.text(0.12, 0.88, "vbt-sensitive\n(signal-driven)", transform=ax.transAxes, fontsize=6, ha="center", color="#3C5488", alpha=0.7)
    ax.text(0.12, 0.08, "engines agree", transform=ax.transAxes, fontsize=6, ha="center", color="#00A087", alpha=0.7)

    r_val, _ = spearmanr(bt, vbt)
    ax.text(0.97, 0.97, f"Spearman r = {r_val:.2f}", transform=ax.transAxes, fontsize=7, ha="right", va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCC", lw=0.5))
    ax.set_xlabel(f"{pair_label(col_bt)} divergence (log-scaled)", fontsize=8)
    ax.set_ylabel(f"{pair_label(col_vbt)} divergence (log-scaled)", fontsize=8)
    ax.set_title("engine failure modes are uncorrelated", fontsize=9)
    add_category_legend(ax)


def _draw_driver_heatmap(ax, sub) -> None:
    features = ["total_cost", "per_trade_cost", "num_trades", "is_ml", "ann_volatility_pct"]
    feat_labels = ["total cost", "cost/trade", "trades", "ML signal", "volatility"]
    engines = sorted(c for c in sub.columns if c.startswith("div_") and c.endswith("_max_rel_pct"))
    eng_labels = [pair_label(c) + " div." for c in engines]

    corr = np.zeros((len(engines), len(features)))
    pvals = np.zeros_like(corr)
    for i, eng in enumerate(engines):
        for j, feat in enumerate(features):
            vals = sub[[eng, feat]].dropna()
            if len(vals) >= 4:
                rho, p = spearmanr(vals.iloc[:, 0], vals.iloc[:, 1])
                corr[i, j], pvals[i, j] = rho, p

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            sig = "+" if pvals[i, j] < 0.05 else ""
            tc = "white" if abs(corr[i, j]) > 0.55 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}{sig}", ha="center", va="center", fontsize=8, color=tc, fontweight="bold" if sig else "normal")

    ax.set_yticks(range(len(eng_labels)))
    ax.set_yticklabels(eng_labels, fontsize=7)
    ax.set_xticks(range(len(feat_labels)))
    ax.set_xticklabels(feat_labels, fontsize=7, rotation=25, ha="right")
    ax.set_title("divergence driver fingerprint", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Spearman rho")


def plot_metric_sensitivity(df: pd.DataFrame, conc: pd.DataFrame | None = None) -> None:
    """ES distribution across benchmarks for each metric (violin + strip)."""
    import seaborn as sns

    if conc is None:
        conc = compute_engine_concordance()
    if conc.empty:
        return

    es_cols = [c for c in conc.columns if c.startswith("es_")]
    if len(es_cols) < 2:
        return

    long = conc[es_cols].melt(var_name="metric", value_name="es")
    long["metric"] = long["metric"].str.replace("es_", "").map(lambda x: _METRIC_LABELS.get(x, x.replace("_pct", "").replace("_", " ")))
    long["bid"] = np.tile(conc.index.values, len(es_cols))
    long["cat"] = long["bid"].map(category)
    n = get_n_buckets(df)

    p2, p98 = long["es"].quantile(0.02), long["es"].quantile(0.98)
    pad = (p98 - p2) * 0.15
    xlim = (max(0, p2 - pad), p98 + pad)

    with pub_style():
        fig, ax = plt.subplots(figsize=(5.5, 3), layout="constrained")
        order = long.groupby("metric")["es"].median().sort_values(ascending=False).index.tolist()
        sns.violinplot(data=long, y="metric", x="es", order=order, color="#E8E8E8", linewidth=0.5, inner=None, ax=ax, cut=0)
        sns.stripplot(data=long, y="metric", x="es", hue="cat", palette=CATEGORY_COLORS, order=order, size=4, alpha=0.7, legend=False, jitter=0.12, ax=ax, zorder=5)
        ax.axvline(1.0, color="#AAAAAA", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_xlim(*xlim)
        ax.set_xlabel("engine sensitivity ES (%)", fontsize=8)
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", linewidth=0.2, alpha=0.3)
        ax.set_title("metric sensitivity to engine choice (rank rho > 0.99 across all pairs)", fontsize=9)
        if n:
            add_n_label(ax, n)
        save(fig, "metric-sensitivity.png", subdir="mechanism")


def plot_complexity_analysis(df: pd.DataFrame) -> None:
    """multi-panel: divergence vs trade count, per-trade cost, and bt-vbt asymmetry."""
    div_cols = [c for c in df.columns if c.startswith("div_") and c.endswith("_max_rel_pct")]
    col_bt = _find_pair_col(div_cols, "bt", "ours")
    col_vbt = _find_pair_col(div_cols, "ours", "vectorbt")
    needed = {"num_trades", "total_commissions", "total_slippage"}
    if not needed.issubset(df.columns) or not col_bt or not col_vbt:
        return

    sub = df.dropna(subset=[col_bt, col_vbt] + list(needed)).copy()
    sub["total_cost"] = sub["total_commissions"] + sub["total_slippage"]
    sub["per_trade_cost"] = sub["total_cost"] / sub["num_trades"].replace(0, np.nan)
    n = get_n_buckets(df)

    bt_label = pair_label(col_bt) + " div. (%)"
    vbt_label = pair_label(col_vbt) + " div. (%)"

    with pub_style():
        fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), layout="constrained")

        _scatter_panel(axes[0], sub, "num_trades", col_bt, "trade count", bt_label)
        _scatter_panel(axes[1], sub, "per_trade_cost", col_bt, "cost per trade ($)", bt_label)
        _scatter_panel(axes[2], sub, col_bt, col_vbt, bt_label, vbt_label)

        _add_diagonal(axes[2], sub, col_bt, col_vbt)

        fig.suptitle("strategy complexity vs divergence (not just commissions)", fontsize=9)
        if n:
            add_n_label(axes[-1], n)
        save(fig, "complexity-analysis.png", subdir="mechanism")


def _scatter_panel(ax, sub, xcol, ycol, xlabel, ylabel) -> None:
    for bid, row in sub.iterrows():
        c = bid_color(bid)
        ax.scatter(row[xcol], row[ycol], color=c, s=30, zorder=5, edgecolors="white", linewidths=0.4, alpha=0.85)
        ax.annotate(short_label(bid), (row[xcol], row[ycol]), fontsize=5,
                    xytext=(3, 3), textcoords="offset points", color=c,
                    clip_on=True, annotation_clip=True)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(linewidth=0.2, alpha=0.3)


def _add_diagonal(ax, sub, col_x, col_y) -> None:
    """draw y=x diagonal reference line."""
    lim = max(sub[col_x].max(), sub[col_y].max()) * 1.1
    ax.plot([0, lim], [0, lim], "--", color="#AAA", linewidth=0.7)
    ax.text(lim * 0.7, lim * 0.5, "x = y", fontsize=6, color="#888", rotation=35)
