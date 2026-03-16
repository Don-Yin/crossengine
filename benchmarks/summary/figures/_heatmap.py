"""heatmap figures: performance dashboard, divergence correlation, engine concordance."""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summary.figures._common import (
    CMAP_NATURE_REDS, add_n_label, annotate_cells, apply_label_colors,
    bid_color, get_n_buckets, get_pop_std, labels_and_colors,
    make_col_labels, normalize_cols, pub_style, save, short_label,
)

def plot_performance_heatmap(df: pd.DataFrame) -> None:
    """annotated heatmap with mean +/- std per cell across buckets."""
    cols = ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "ann_volatility_pct",
            "max_drawdown_pct", "cagr_pct"]
    lower_better = {"ann_volatility_pct"}
    present = [c for c in cols if c in df.columns]
    if len(present) < 3:
        return

    sub = df[present].dropna()
    labels, label_colors = labels_and_colors(sub.index)
    normed = normalize_cols(sub, lower_better)
    col_labels = make_col_labels(present, lower_better)
    n = get_n_buckets(df)

    std_data = {}
    for c in present:
        std_data[c] = get_pop_std(df, c).reindex(sub.index).fillna(0) if f"pop_std_{c}" in df.columns else pd.Series(0, index=sub.index)

    with pub_style():
        fig, ax = plt.subplots(
            figsize=(0.80 * len(present) + 1.3, 0.23 * len(labels) + 0.7),
            layout="constrained",
        )
        im = ax.imshow(normed.values, aspect="auto", cmap=CMAP_NATURE_REDS)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(col_labels, fontsize=7, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)))
        ylabels = ax.set_yticklabels(labels, fontsize=7)
        apply_label_colors(ylabels, label_colors)

        _annotate_with_std(ax, sub, normed, std_data, present)

        fig.colorbar(im, ax=ax, shrink=0.6, label="normalized (darker = better)")
        if n:
            add_n_label(ax, n)
        save(fig, "performance-heatmap.png", subdir="performance")


def _annotate_with_std(ax, sub, normed, std_data, present) -> None:
    """write mean +/- std text into heatmap cells."""
    for i in range(len(sub)):
        for j, c in enumerate(present):
            val = sub.iloc[i, j]
            std = std_data[c].iloc[i] if hasattr(std_data[c], "iloc") else 0
            if std > 0.005:
                txt = f"{val:.1f}\n+/-{std:.1f}"
            else:
                txt = f"{val:.2f}"
            text_color = "white" if normed.iloc[i, j] > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=4.5, color=text_color)


def plot_divergence_correlation(df: pd.DataFrame) -> None:
    """pairwise correlation of daily engine divergence (ours - bt) across benchmarks."""
    from summary.collect import collect_divergence_correlations
    corr = collect_divergence_correlations()
    if corr.empty:
        return

    labels, label_colors = labels_and_colors(corr.index)
    n = get_n_buckets(df)

    with pub_style():
        fig, ax = plt.subplots(
            figsize=(0.4 * len(labels) + 1.5, 0.35 * len(labels) + 1),
            layout="constrained",
        )
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(labels)))
        xlabels = ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(labels)))
        ylabels = ax.set_yticklabels(labels, fontsize=6)
        apply_label_colors(xlabels, label_colors)
        apply_label_colors(ylabels, label_colors)

        def fmt(val, _):
            return f"{val:.2f}"
        annotate_cells(ax, corr.values, np.abs(corr.values), fmt, threshold=0.6)

        fig.colorbar(im, ax=ax, shrink=0.6, label="divergence correlation")
        if n:
            add_n_label(ax, n)
        save(fig, "divergence-correlation.png", subdir="divergence")


def plot_engine_concordance(df: pd.DataFrame, conc: pd.DataFrame | None = None) -> None:
    """heatmap of engine sensitivity (ES) per benchmark per metric."""
    if conc is None:
        from summary.concordance import compute_engine_concordance
        conc = compute_engine_concordance()
    if conc.empty:
        return

    es_cols = [c for c in conc.columns if c.startswith("es_")]
    if len(es_cols) < 2:
        return

    sub = conc[es_cols].copy()
    sub.columns = [c.replace("es_", "").replace("_pct", " (%)").replace("_", " ")
                   for c in sub.columns]
    labels, label_colors = labels_and_colors(sub.index)
    n = get_n_buckets(df)

    with pub_style():
        fig, ax = plt.subplots(
            figsize=(0.80 * len(sub.columns) + 1.3, 0.23 * len(labels) + 0.7),
            layout="constrained",
        )
        im = ax.imshow(sub.values, aspect="auto", cmap=CMAP_NATURE_REDS)
        ax.set_xticks(range(len(sub.columns)))
        ax.set_xticklabels(sub.columns, fontsize=7, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)))
        ylabels = ax.set_yticklabels(labels, fontsize=7)
        apply_label_colors(ylabels, label_colors)

        nmax = sub.values.max()
        normed = sub.values / nmax if nmax > 0 else sub.values * 0

        def fmt(val, _):
            return f"{val:.2f}%"
        annotate_cells(ax, sub.values, normed, fmt, threshold=0.5)

        fig.colorbar(im, ax=ax, shrink=0.6, label="engine sensitivity ES (%)")
        if n:
            add_n_label(ax, n)
        save(fig, "engine-concordance.png", subdir="divergence")


def plot_engine_agreement(df: pd.DataFrame, conc: pd.DataFrame | None = None) -> None:
    """per-bucket engine difference distributions (strip plot)."""
    from summary.concordance import collect_bucket_engine_stats
    diffs = _engine_diffs(collect_bucket_engine_stats())
    if diffs.empty:
        return
    metrics = ["total_return_pct", "sharpe", "max_dd_pct"]
    mlabels = {"total_return_pct": "total ret diff (%)",
               "sharpe": "Sharpe diff", "max_dd_pct": "max dd diff (%)"}
    bids = sorted(diffs["benchmark_id"].unique())
    labels = [short_label(b) for b in bids]
    diffs["label"] = diffs["benchmark_id"].map(dict(zip(bids, labels)))
    with pub_style():
        fig, axes = plt.subplots(1, len(metrics),
                                 figsize=(3.0 * len(metrics), 0.26 * len(bids) + 1.2),
                                 sharey=True)
        fig.subplots_adjust(wspace=0.25, top=0.92, bottom=0.14, left=0.18, right=0.98)
        for i, (ax, mk) in enumerate(zip(axes, metrics)):
            mdata = diffs[diffs["metric"] == mk]
            if not mdata.empty:
                _draw_diff_strip(ax, mdata, labels, mlabels[mk])
        axes[0].set_yticks(range(len(labels)))
        ylbl = axes[0].set_yticklabels(labels, fontsize=7)
        apply_label_colors(ylbl, [bid_color(b) for b in bids])
        handles, leg_labels = axes[0].get_legend_handles_labels()
        if handles:
            ncol = min(4, len(leg_labels))
            fig.legend(handles, leg_labels, loc="lower center",
                       ncol=ncol, fontsize=5.5, frameon=True,
                       framealpha=0.9, edgecolor="#CCC",
                       columnspacing=1.0, handletextpad=0.4,
                       bbox_to_anchor=(0.58, -0.04))
        fig.suptitle(f"engine agreement (n = {diffs['bucket_id'].nunique()} buckets)",
                     fontsize=9, y=0.97)
        save(fig, "engine-agreement.png", tight=True, subdir="divergence")


def _engine_diffs(long: pd.DataFrame) -> pd.DataFrame:
    """pivot per-engine stats to all pairwise differences."""
    wide = long.pivot_table(index=["benchmark_id", "bucket_id", "metric"],
                            columns="engine", values="value").reset_index()
    meta = {"benchmark_id", "bucket_id", "metric"}
    engines = sorted(c for c in wide.columns if c not in meta)
    parts = []
    for a, b in itertools.combinations(engines, 2):
        tmp = wide[list(meta)].copy()
        tmp["comparison"], tmp["diff"] = f"{a} - {b}", wide[a] - wide[b]
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _draw_diff_strip(ax, mdata, labels, xlabel) -> None:
    """boxplot of engine differences for one metric (compact, shows IQR not min/max)."""
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter
    from summary.figures._common import pair_color
    comparisons = sorted(mdata["comparison"].unique())
    pal = {c: pair_color(c.replace(" - ", "_vs_")) for c in comparisons}
    sns.boxplot(data=mdata, y="label", x="diff", hue="comparison",
                palette=pal, order=labels, hue_order=comparisons,
                linewidth=0.6, fliersize=1.5, whis=1.5, ax=ax, legend=True)
    if ax.get_legend():
        ax.get_legend().remove()
    ax.axvline(0, color="#AAAAAA", linewidth=0.7, linestyle="--", zorder=0)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(_compact_fmt))
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=5, rotation=35)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="x", linewidth=0.25, alpha=0.35)


def _compact_fmt(x, _pos) -> str:
    """format symlog tick values without scientific notation."""
    if x == 0:
        return "0"
    ax_val = abs(x)
    if ax_val >= 1:
        return f"{x:.0f}"
    if ax_val >= 0.01:
        return f"{x:.2f}"
    return f"{x:.1e}"


from summary.figures._risk_return import plot_risk_return  # noqa: F401 -- re-exported
