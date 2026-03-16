"""divergence-related summary figures: landscape heatmap, scatter, tiered timeseries."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summary.collect import load_all_equity
from summary.figures._common import (
    CATEGORY_COLORS, CATEGORY_TITLES, CMAP_NATURE_BLUES, CMAP_NATURE_REDS,
    add_category_legend, add_n_label, bid_color, bid_style,
    apply_label_colors, get_n_buckets, get_pop_std,
    group_by_category, labels_and_colors, normalize_cols, pair_label,
    pub_style, save, short_label,
)


def plot_divergence_landscape(df) -> None:
    """dual-zone heatmap: divergence outcome (left) vs strategy complexity (right)."""
    div_cols = sorted(c for c in df.columns if c.startswith("div_") and c.endswith("_max_rel_pct"))
    if len(div_cols) < 1:
        return

    sub = df.dropna(subset=div_cols).copy()
    sub["cost_bps"] = (sub.get("total_commissions", 0) + sub.get("total_slippage", 0)) / 100_000 * 10_000
    feat_cols = ["num_trades", "cost_bps", "ann_volatility_pct"]
    for c in feat_cols:
        if c not in sub.columns:
            sub[c] = 0

    order = sub[div_cols].max(axis=1).sort_values(ascending=False).index
    sub_div = sub.loc[order, div_cols]
    sub_feat = sub.loc[order, feat_cols]
    norm_div = normalize_cols(sub_div)
    norm_feat = normalize_cols(sub_feat)
    labels, lcols = labels_and_colors(order)
    n = get_n_buckets(df)

    with pub_style():
        fig = plt.figure(figsize=(7.5, 0.23 * len(order) + 1.0), layout="constrained")
        gs = gridspec.GridSpec(1, 2, width_ratios=[len(div_cols), len(feat_cols)],
                               wspace=0.06, figure=fig)
        ax_d = fig.add_subplot(gs[0])
        ax_f = fig.add_subplot(gs[1])

        ax_d.imshow(norm_div.values, cmap=CMAP_NATURE_REDS, aspect="auto", vmin=0, vmax=1)
        ax_f.imshow(norm_feat.values, cmap=CMAP_NATURE_BLUES, aspect="auto", vmin=0, vmax=1)

        _annotate_zone(ax_d, sub_div.values, norm_div.values, is_pct=True)
        _annotate_zone(ax_f, sub_feat.values, norm_feat.values, pct_cols={2})

        ax_d.set_yticks(range(len(labels)))
        ylbl = ax_d.set_yticklabels(labels, fontsize=7)
        apply_label_colors(ylbl, lcols)
        ax_f.set_yticks([])

        div_labels = [pair_label(c) + " (%)" for c in div_cols]
        ax_d.set_xticks(range(len(div_cols)))
        ax_d.set_xticklabels(div_labels, fontsize=6, rotation=45, ha="right")
        ax_f.set_xticks(range(len(feat_cols)))
        ax_f.set_xticklabels(["trades", "cost (bps)", "vol (%)"], fontsize=7, rotation=45, ha="right")
        ax_d.set_title("divergence", fontsize=9, fontweight="bold")
        ax_f.set_title("strategy profile", fontsize=9, fontweight="bold")
        if n:
            add_n_label(ax_f, n)
        save(fig, "divergence-landscape.png", subdir="divergence")


def _annotate_zone(ax, vals, normed, is_pct=False, pct_cols=None) -> None:
    pct_cols = pct_cols or set()
    nr, nc = vals.shape
    for i in range(nr):
        for j in range(nc):
            v = vals[i, j]
            if is_pct or j in pct_cols:
                txt = f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"
            else:
                txt = f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}"
            tc = "white" if normed[i, j] > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.5, color=tc)


def plot_divergence_vs_cost(df) -> None:
    """scatter with error crosses showing +/-1 std in both axes."""
    div_cols = [c for c in df.columns if c.startswith("div_") and c.endswith("_max_rel_pct")]
    needed = {"total_commissions", "total_slippage"}
    if not needed.issubset(df.columns) or not div_cols:
        return

    sub = df.dropna(subset=list(needed) + div_cols).copy()
    sub["max_div"] = sub[div_cols].max(axis=1)
    sub["total_cost"] = sub["total_commissions"] + sub["total_slippage"]
    n = get_n_buckets(df)

    comm_std = get_pop_std(df, "total_commissions")
    slip_std = get_pop_std(df, "total_slippage")
    cost_std = np.sqrt(comm_std**2 + slip_std**2).reindex(sub.index).fillna(0)

    with pub_style():
        fig, ax = plt.subplots(figsize=(5.5, 4), layout="constrained")
        for bid, row in sub.iterrows():
            color = bid_color(bid)
            x_err = cost_std.loc[bid] if hasattr(cost_std, "loc") else 0
            ax.errorbar(row["total_cost"], row["max_div"],
                        xerr=x_err, yerr=0,
                        fmt="o", color=color, markersize=4.5, zorder=5,
                        markeredgecolor="white", markeredgewidth=0.4,
                        elinewidth=0.6, capsize=2, capthick=0.4, alpha=0.85)
            ax.annotate(short_label(bid), (row["total_cost"], row["max_div"]),
                        fontsize=5.5, xytext=(4, 3), textcoords="offset points", color=color)

        _add_trend_line(ax, sub)
        ax.set_xlabel("total transaction costs ($)")
        ax.set_ylabel("max relative divergence across pairs (%)")
        add_category_legend(ax)
        if n:
            add_n_label(ax, n)
        save(fig, "divergence-vs-cost.png", subdir="divergence")


def _add_trend_line(ax, sub) -> None:
    if len(sub) <= 2:
        return
    from numpy.polynomial.polynomial import polyfit
    x, y = sub["total_cost"].values, sub["max_div"].values
    mask = x > 0
    if mask.sum() <= 2:
        return
    c = polyfit(np.log1p(x[mask]), y[mask], 1)
    xs = np.linspace(x[mask].min(), x[mask].max(), 100)
    ys = np.maximum(c[0] + c[1] * np.log1p(xs), 0)
    ax.plot(xs, ys, "--", color="#333", linewidth=0.8, alpha=0.6)


def plot_divergence_timeseries(df) -> None:
    """relative divergence (%) faceted by strategy category (rows) x engine (cols)."""
    equities = load_all_equity()
    if not equities:
        return

    valid = sorted(bid for bid in equities if "ours" in equities[bid].columns)
    groups = group_by_category(valid)
    if not groups:
        return
    n = get_n_buckets(df)
    sample_eq = next(iter(equities.values()), pd.DataFrame())
    refs = sorted(c for c in sample_eq.columns if c != "ours")
    n_refs = len(refs)
    n_cats = len(groups)

    with pub_style():
        fig, axes = plt.subplots(
            n_cats, n_refs,
            figsize=(3.6 * n_refs, 1.8 * n_cats + 0.8),
            sharex=True, squeeze=False,
            gridspec_kw={"hspace": 0.35, "wspace": 0.12},
        )

        for ri, (cat, bids) in enumerate(groups.items()):
            for ci, ref in enumerate(refs):
                ax = axes[ri, ci]
                _plot_category_panel(ax, equities, bids, ref)
                if ci == 0:
                    ax.set_ylabel(CATEGORY_TITLES.get(cat, cat), fontsize=7,
                                  color=CATEGORY_COLORS.get(cat, "#333"))
                else:
                    ax.tick_params(labelleft=False)
                if ri == 0:
                    ax.set_title(f"vs {ref}", fontsize=8, fontweight="bold")
            for ci in range(1, n_refs):
                axes[ri, ci].sharey(axes[ri, 0])

        axes[-1, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate(rotation=30, ha="right")
        if n:
            add_n_label(axes[-1, -1], n)
        save(fig, "divergence-timeseries.png", subdir="divergence")



def _plot_category_panel(ax, equities, bids, ref) -> None:
    for bid in bids:
        eq = equities.get(bid)
        if eq is None or ref not in eq.columns:
            continue
        midpoint = ((eq["ours"] + eq[ref]) / 2).replace(0, np.nan)
        rel = (eq["ours"] - eq[ref]) / midpoint * 100
        ax.plot(rel.index, rel.values, color=bid_color(bid),
                linestyle=bid_style(bid), linewidth=0.9,
                label=short_label(bid), alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_ylabel("")
    ax.legend(loc="best", fontsize=5, ncol=1)
    ax.grid(axis="y", linewidth=0.2, alpha=0.3)
