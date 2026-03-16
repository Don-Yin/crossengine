"""distribution-based visual tables with Nature-grade box + strip styling."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from summary.bucket_data import collect_bucket_metrics
from summary.figures._common import bid_color, pub_style, save, short_label
from summary.figures._significance import annotate_significance, compute_significance

_BLUE = "#3C5488"
_ROSE = "#E64B35"
_CYAN = "#91D1C2"
_DOT = "#3C5488"

_BG_TOP = np.array([0.88, 0.96, 0.92])
_BG_BOT = np.array([0.98, 0.90, 0.87])


def _rank_color(pct: float) -> tuple:
    """map rank percentile (0=best, 1=worst) to subtle background rgb."""
    mid = np.ones(3)
    anchor, target = (_BG_TOP, mid) if pct <= 0.5 else (mid, _BG_BOT)
    t = pct * 2 if pct <= 0.5 else (pct - 0.5) * 2
    return tuple(anchor * (1 - t) + target * t)


# -- data preparation -----------------------------------------------------------


def _prep_data(long: pd.DataFrame, metrics: list[str]) -> tuple:
    """filter, label, return (sub, bids, labels, n_buckets)."""
    sub = long[long["metric"].isin(metrics)].copy()
    if sub.empty:
        return sub, [], [], 0
    bids = sorted(sub["benchmark_id"].unique())
    labels = [short_label(b) for b in bids]
    sub["label"] = sub["benchmark_id"].map(dict(zip(bids, labels)))
    n_buckets = int(sub.groupby("benchmark_id")["bucket_id"].nunique().max())
    return sub, bids, labels, n_buckets


def _make_fig(n_metrics: int, n_bids: int):
    """create figure and flat axes list."""
    w = max(2.0 * n_metrics + 0.5, 6)
    h = 0.32 * n_bids + 1.6
    fig, axes = plt.subplots(1, n_metrics, figsize=(w, h), sharey=True, layout="constrained")
    return fig, [axes] if n_metrics == 1 else list(axes)


# -- per-subplot rendering ------------------------------------------------------


def _shade_rows(ax, labels: list[str], mdata: pd.DataFrame, lower_better: bool) -> None:
    """shade row backgrounds by rank of median value."""
    medians = mdata.groupby("label")["value"].median().reindex(labels)
    ranks = medians.rank(ascending=lower_better, pct=True).fillna(0.5)
    for idx, lbl in enumerate(labels):
        ax.axhspan(idx - 0.5, idx + 0.5, color=_rank_color(ranks[lbl]), zorder=0)


def _draw_single(ax, mdata: pd.DataFrame, labels: list[str]) -> None:
    """box + strip for single-distribution metrics."""
    sns.boxplot(
        data=mdata,
        y="label",
        x="value",
        color=_CYAN,
        ax=ax,
        orient="h",
        linewidth=0.6,
        fliersize=0,
        width=0.5,
        order=labels,
        boxprops=dict(alpha=0.55),
        medianprops=dict(color="#333", linewidth=0.8),
    )
    sns.stripplot(data=mdata, y="label", x="value", color=_DOT, ax=ax, orient="h", size=1.6, alpha=0.4, order=labels, jitter=0.12, legend=False)


def _draw_split(ax, mdata: pd.DataFrame, labels: list[str]) -> None:
    """box + strip for two-group (SPX vs asset_avg) distributions."""
    pal = {"spx": _BLUE, "asset_avg": _ROSE}
    sns.boxplot(
        data=mdata,
        y="label",
        x="value",
        hue="section",
        palette=pal,
        ax=ax,
        orient="h",
        linewidth=0.5,
        fliersize=0,
        width=0.65,
        order=labels,
        boxprops=dict(alpha=0.5),
        medianprops=dict(color="#333", linewidth=0.7),
    )
    sns.stripplot(data=mdata, y="label", x="value", hue="section", palette=pal, ax=ax, orient="h", size=1.2, alpha=0.35, dodge=True, order=labels, jitter=0.08, legend=False)


def _style_ax(ax, label: str, ref_line: float | None) -> None:
    """axis cosmetics: label, gridlines, optional reference line."""
    if ref_line is not None:
        ax.axvline(ref_line, color="#AAAAAA", linewidth=0.6, linestyle="--", zorder=1)
    ax.set_xlabel(label, fontsize=7)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=6.5)
    ax.tick_params(axis="x", labelsize=5.5)
    ax.grid(axis="x", linewidth=0.25, alpha=0.35)


def _clamp_xlim(ax, mdata: pd.DataFrame, ref: float | None = None) -> None:
    """set robust x-axis limits using P2/P98 to avoid extreme outlier stretching."""
    vals = mdata["value"].dropna()
    if vals.empty:
        return
    lo, hi = np.percentile(vals, [2, 98])
    pad = max((hi - lo) * 0.15, 1e-6)
    lo, hi = lo - pad, hi + pad
    if ref is not None:
        lo, hi = min(lo, ref - pad), max(hi, ref + pad)
    ax.set_xlim(lo, hi)


def _color_ticks(axes, bids: list[str]) -> None:
    """color y-tick labels to match benchmark palette."""
    ticks = axes[0].get_yticklabels()
    for i, bid in enumerate(bids):
        if i < len(ticks):
            ticks[i].set_color(bid_color(bid))


# -- generic distribution renderer ----------------------------------------------


def _plot_dist(
    long: pd.DataFrame,
    metrics: list[str],
    labels: dict[str, str],
    lower: set[str],
    refs: dict[str, float],
    title: str,
    filename: str,
    split: bool = False,
) -> None:
    """core distribution figure: one subplot per metric, rows = benchmarks."""
    sub, bids, lbls, nb = _prep_data(long, metrics)
    if len(bids) < 2:
        return
    present = [m for m in metrics if m in sub["metric"].unique()]
    sig_map = compute_significance(sub, present, split)
    with pub_style():
        fig, axes = _make_fig(len(present), len(bids))
        for i, (ax, metric) in enumerate(zip(axes, present)):
            mdata = sub[sub["metric"] == metric]
            _shade_rows(ax, lbls, mdata, metric in lower)
            (_draw_split if split else _draw_single)(ax, mdata, lbls)
            arrow = " <-" if metric in lower else " ->"
            _style_ax(ax, labels.get(metric, metric) + arrow, refs.get(metric))
            _clamp_xlim(ax, mdata, refs.get(metric))
            _handle_legend(ax, split, i == 0)
            annotate_significance(ax, metric, lbls, sig_map, split)
        _color_ticks(axes, bids)
        fig.suptitle(f"{title}  (n = {nb} buckets)", fontsize=9, y=1.02)
        if sig_map:
            fig.text(0.99, -0.01, "+ p<.05  ++ p<.01  +++ p<.001  (BH-FDR)", fontsize=5, ha="right", va="top", color="#888")
        save(fig, filename, subdir="performance")


def _handle_legend(ax, split: bool, is_first: bool) -> None:
    """keep a compact legend only on the first split subplot."""
    if split and is_first and ax.get_legend():
        h, _ = ax.get_legend_handles_labels()
        ax.get_figure().legend(h[:2], ["spx", "asset avg"], fontsize=5.5, title="control", title_fontsize=5.5, loc="upper right", framealpha=0.9, bbox_to_anchor=(1.0, 1.06))
        ax.get_legend().remove()
    elif ax.get_legend():
        ax.legend().remove()


# -- metric definitions ----------------------------------------------------------

_PERF = ["cagr_pct", "ann_volatility_pct", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown_pct", "win_rate_pct"]
_PERF_LBL = {
    "cagr_pct": "cagr (%)",
    "ann_volatility_pct": "volatility (%)",
    "sharpe_ratio": "sharpe",
    "sortino_ratio": "sortino",
    "calmar_ratio": "calmar",
    "max_drawdown_pct": "max dd (%)",
    "win_rate_pct": "win rate (%)",
}
_PERF_LOW = {"ann_volatility_pct"}
_PERF_REF: dict[str, float] = {"sharpe_ratio": 0, "sortino_ratio": 0, "calmar_ratio": 0}

_COST = ["num_trades", "total_commissions", "total_slippage", "div_bt_max_abs_diff", "div_bt_max_rel_diff_pct", "div_vectorbt_max_abs_diff", "div_vectorbt_max_rel_diff_pct"]
_COST_LBL = {
    "num_trades": "trades",
    "total_commissions": "commissions ($)",
    "total_slippage": "slippage ($)",
    "div_bt_max_abs_diff": "bt max abs ($)",
    "div_bt_max_rel_diff_pct": "bt max rel (%)",
    "div_vectorbt_max_abs_diff": "vbt max abs ($)",
    "div_vectorbt_max_rel_diff_pct": "vbt max rel (%)",
}
_COST_LOW = {"total_commissions", "total_slippage", "div_bt_max_abs_diff", "div_bt_max_rel_diff_pct", "div_vectorbt_max_abs_diff", "div_vectorbt_max_rel_diff_pct"}
_COST_REF: dict[str, float] = {}

_CTRL = ["excess_return_pct", "alpha_ann_pct", "beta", "information_ratio", "tracking_error_pct", "up_capture", "down_capture"]
_CTRL_LBL = {
    "excess_return_pct": "excess return (%)",
    "alpha_ann_pct": "alpha (%)",
    "beta": "beta",
    "information_ratio": "info ratio",
    "tracking_error_pct": "tracking err (%)",
    "up_capture": "up capture",
    "down_capture": "down capture",
}
_CTRL_LOW = {"tracking_error_pct", "down_capture"}
_CTRL_REF: dict[str, float] = {
    "excess_return_pct": 0,
    "alpha_ann_pct": 0,
    "beta": 1,
    "information_ratio": 0,
    "up_capture": 1,
    "down_capture": 1,
}


# -- public API ------------------------------------------------------------------


def plot_table_raw(df: pd.DataFrame) -> None:
    """performance distribution across buckets for each benchmark."""
    long = collect_bucket_metrics(sections=("engine",))
    _plot_dist(long, _PERF, _PERF_LBL, _PERF_LOW, _PERF_REF, "performance distribution", "table-performance.png")


def plot_controlled_distributions(df: pd.DataFrame) -> None:
    """SPX vs asset-avg controlled metric distributions."""
    long = collect_bucket_metrics(sections=("spx", "asset_avg"))
    _plot_dist(long, _CTRL, _CTRL_LBL, _CTRL_LOW, _CTRL_REF, "controlled metrics -- SPX vs asset average", "controlled-distributions.png", split=True)


def plot_table_costs(df: pd.DataFrame) -> None:
    """cost and divergence distribution across buckets."""
    long = collect_bucket_metrics(sections=("engine", "div_bt", "div_vectorbt"))
    _plot_dist(long, _COST, _COST_LBL, _COST_LOW, _COST_REF, "costs and engine divergence", "table-costs-divergence.png")
