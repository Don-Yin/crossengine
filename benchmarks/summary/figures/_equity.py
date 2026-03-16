"""equity-curve and drawdown summary figures."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summary.collect import load_all_equity
from summary.figures._common import (
    CATEGORY_COLORS, CATEGORY_TITLES, add_n_label, bid_color, bid_style,
    category, get_n_buckets, group_by_category, pub_style, save, short_label,
)
from utils.comparison import ENGINE_COLORS, ENGINE_LINESTYLES, ENGINE_LINEWIDTHS  # noqa: F401
from utils.data import RESULTS_ROOT


def _load_equity_std(bid: str) -> pd.DataFrame | None:
    """load equity_std.csv (bucket-to-bucket std) for one benchmark."""
    p = RESULTS_ROOT / bid / "equity_std.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0, parse_dates=True)


def plot_equity_overlay(df: pd.DataFrame) -> None:
    """strategy equity curves rebased to 100, faceted by category, with +/-1 std bands."""
    equities = load_all_equity()
    if not equities:
        return

    from utils.data import load_spx
    spx = load_spx()
    n = get_n_buckets(df)

    valid = {bid for bid, edf in equities.items() if "ours" in edf.columns}
    groups = group_by_category(list(valid))
    n_cats = len(groups)
    if n_cats == 0:
        return

    all_rebased = [equities[bid]["ours"].pipe(lambda v: v / v.iloc[0] * 100) for bid in valid]
    spx_reb = _rebase_spx(spx, equities, valid)
    if spx_reb is not None:
        all_rebased.append(spx_reb)

    ymin, ymax = min(s.min() for s in all_rebased), max(s.max() for s in all_rebased)
    pad = (ymax - ymin) * 0.05
    ylims = (ymin - pad, ymax + pad)

    with pub_style():
        fig, axes = plt.subplots(n_cats, 1, figsize=(12, 3.0 * n_cats + 1.0),
                                 sharex=True, layout="constrained")
        if n_cats == 1:
            axes = [axes]

        for ax, (cat, bids) in zip(axes, groups.items()):
            _plot_equity_category(ax, equities, bids, cat, spx_reb, ylims)

        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate(rotation=30, ha="right")
        if n:
            add_n_label(axes[-1], n)
        save(fig, "equity-overlay.png", subdir="performance")


def _rebase_spx(spx, equities, valid) -> pd.Series | None:
    if spx is None or len(spx) == 0:
        return None
    first_eq = equities[min(valid)]
    common = spx.index.intersection(first_eq.index)
    if len(common) == 0:
        return None
    spx_c = spx.loc[common]
    return spx_c / spx_c.iloc[0] * 100


def _plot_equity_category(ax, equities, bids, cat, spx_reb, ylims) -> None:
    if spx_reb is not None:
        ax.plot(spx_reb.index, spx_reb.values, color="#B0B0B0",
                linewidth=1.0, linestyle="--", label="S&P 500", alpha=0.7, zorder=0)
    for bid in bids:
        v = equities[bid]["ours"]
        reb = v / v.iloc[0] * 100
        color = bid_color(bid)
        ax.plot(reb.index, reb.values, color=color, linestyle=bid_style(bid),
                linewidth=1.0, label=short_label(bid), alpha=0.85)
        _add_std_band(ax, bid, v, reb, color)
    ax.axhline(100, color="black", linewidth=0.3, linestyle=":")
    ax.set_ylim(ylims)
    ax.set_ylabel("rebased (100)", fontsize=7)
    ax.set_title(CATEGORY_TITLES[cat], fontsize=8, loc="left")
    ax.legend(loc="upper left", fontsize=6, ncol=2)


def _add_std_band(ax, bid, v, reb, color) -> None:
    std_df = _load_equity_std(bid)
    if std_df is None or "ours" not in std_df.columns:
        return
    std_reb = std_df["ours"] / v.iloc[0] * 100
    common_idx = reb.index.intersection(std_reb.index)
    lo = np.maximum(reb.loc[common_idx] - std_reb.loc[common_idx], 0)
    hi = reb.loc[common_idx] + std_reb.loc[common_idx]
    ax.fill_between(common_idx, lo, hi, color=color, alpha=0.12)


def _compute_dd(v: pd.Series) -> pd.Series:
    """compute percentage drawdown from equity curve."""
    return (v - v.cummax()) / v.cummax() * 100


def plot_drawdown_comparison(df: pd.DataFrame) -> None:
    """small-multiple 3-engine drawdown overlay, sorted by engine sensitivity."""
    equities = load_all_equity()
    if not equities:
        return

    from summary.concordance import compute_engine_concordance
    conc = compute_engine_concordance()

    valid = sorted(bid for bid, edf in equities.items()
                   if "ours" in edf.columns and len(edf.columns) >= 2)
    if not valid:
        return

    es_col = "es_max_dd_pct"
    if not conc.empty and es_col in conc.columns:
        valid = sorted(valid, key=lambda b: conc.loc[b, es_col] if b in conc.index else 0)

    ncols = min(5, len(valid))
    nrows = -(-len(valid) // ncols)
    n = get_n_buckets(df)

    with pub_style():
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.2 * nrows),
                                 sharex=True, sharey=True, layout="constrained")
        flat = axes.flat if hasattr(axes, "flat") else [axes]

        for i, bid in enumerate(valid):
            ax = flat[i]
            edf = equities[bid]
            _draw_dd_panel(ax, edf, bid, conc, es_col)

        for j in range(len(valid), len(flat)):
            flat[j].set_visible(False)

        flat[0].xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        flat[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate(rotation=30, ha="right")
        fig.suptitle("engine drawdown comparison (sorted by ES)", fontsize=9)
        if n:
            add_n_label(flat[len(valid) - 1], n)
        save(fig, "drawdown-comparison.png", subdir="performance")


def _draw_dd_panel(ax, edf, bid, conc, es_col) -> None:
    """draw 3-engine drawdown overlay with disagreement envelope for one benchmark."""
    dd_curves = {}
    for eng in edf.columns:
        dd_curves[eng] = _compute_dd(edf[eng])

    for eng, dd in dd_curves.items():
        ax.plot(dd.index, dd.values, color=ENGINE_COLORS.get(eng, "#333"),
                linestyle=ENGINE_LINESTYLES.get(eng, "-"),
                linewidth=ENGINE_LINEWIDTHS.get(eng, 1.0), alpha=0.85)

    if len(dd_curves) >= 2:
        stacked = pd.concat(dd_curves.values(), axis=1)
        ax.fill_between(stacked.index, stacked.min(axis=1), stacked.max(axis=1),
                        color="#CCCCCC", alpha=0.25)

    ax.axhline(0, color="black", linewidth=0.3)
    cat = category(bid)
    ax.set_title(short_label(bid), fontsize=6, fontweight="bold",
                 color=CATEGORY_COLORS.get(cat, "#333"))
    ax.set(xlabel="", ylabel="")
    ax.tick_params(labelsize=5)

    if not conc.empty and bid in conc.index and es_col in conc.columns:
        es_val = conc.loc[bid, es_col]
        ax.text(0.97, 0.06, f"ES={es_val:.2f}%", transform=ax.transAxes,
                fontsize=4.5, ha="right", color="#666")
