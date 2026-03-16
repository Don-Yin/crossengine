"""plotting helpers for engine comparison reports."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

logger = logging.getLogger(__name__)

ENGINE_COLORS = {
    "ours": "#3C5488", "bt": "#E64B35", "vectorbt": "#00A087",
    "backtrader": "#F39B7F", "cvxportfolio": "#8491B4",
    "zipline": "#4DBBD5", "nautilus": "#7E6148",
}
ENGINE_LINESTYLES = {
    "ours": "-", "bt": "--", "vectorbt": ":", "backtrader": "-.",
    "cvxportfolio": (0, (3, 1, 1, 1)),
    "zipline": (0, (3, 1, 1, 1)), "nautilus": (0, (5, 2)),
}
ENGINE_LINEWIDTHS = {
    "ours": 1.4, "bt": 1.2, "vectorbt": 1.2,
    "backtrader": 1.2, "cvxportfolio": 1.2, "zipline": 1.2, "nautilus": 1.2,
}


@contextmanager
def pub_style():
    """matplotlib RC context for publication-quality figures."""
    overrides = {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": 0.6,
        "axes.prop_cycle": plt.cycler(color=[
            "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
            "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
        ]),
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
    with plt.rc_context(overrides):
        yield


def _trade_dates(trades_df: pd.DataFrame) -> list:
    """extract sorted unique rebalance dates from trades."""
    if trades_df.empty:
        return []
    return sorted(trades_df["date"].unique())


def _asset_colors(result) -> dict[str, str]:
    """assign a color to each asset in the result."""
    palette = ["#2E5090", "#E67E22", "#27AE60", "#8E44AD", "#C0392B"]
    assets = result.weights().columns.tolist()
    colors = {"cash": "#C8C8C8"}
    for i, a in enumerate(assets):
        colors[a] = palette[i % len(palette)]
    return colors


def _mark_trades(ax, trade_dates) -> None:
    """draw vertical lines at rebalance dates."""
    for d in trade_dates:
        ax.axvline(d, color="#E0E0E0", alpha=0.6, linewidth=0.5)


def _plot_value_breakdown(ax, result, cmap=None) -> None:
    """stacked area: cash on bottom, then each position's dollar value."""
    tv = result.portfolio_value
    cash = result.cash
    w = result.weights()
    assets = w.columns.tolist()
    if cmap is None:
        cmap = _asset_colors(result)
    stack_colors = [cmap["cash"]] + [cmap[a] for a in assets]
    areas = [cash.values]
    labels = ["cash"]
    for col in assets:
        areas.append(w[col].values * tv.values)
        labels.append(col)
    ax.stackplot(tv.index, *areas, labels=labels, colors=stack_colors, alpha=0.70)
    ax.plot(tv.index, tv.values, color="black", linewidth=1.2, label="total")


def _plot_engine_overlay(ax, merged: pd.DataFrame, tdates: list) -> None:
    """overlay all engine equity curves on one axes."""
    dollar_fmt = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    for eng_name in merged.columns:
        ax.plot(
            merged.index, merged[eng_name], label=eng_name,
            linewidth=ENGINE_LINEWIDTHS.get(eng_name, 1.2),
            color=ENGINE_COLORS.get(eng_name, "#333333"),
            linestyle=ENGINE_LINESTYLES.get(eng_name, "-"),
        )
    _mark_trades(ax, tdates)
    tset = set(tdates)
    if "ours" in merged.columns:
        ours_trade = merged["ours"][merged.index.isin(tset)]
        ax.scatter(ours_trade.index, ours_trade.values, color="#A93226", marker="D", s=12, zorder=5, label="rebalance")
    ax.set_ylabel("All engines ($)")
    ax.yaxis.set_major_formatter(dollar_fmt)
    ax.legend(loc="upper left")


def _plot_diff_panel(ax, merged: pd.DataFrame, ref_name: str, tdates: list) -> None:
    """plot ours minus one reference engine."""
    diff = merged["ours"] - merged[ref_name]
    ax.plot(merged.index, diff, color=ENGINE_COLORS.get(ref_name, "#333333"),
            linestyle=ENGINE_LINESTYLES.get(ref_name, "-"), linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.4)
    _mark_trades(ax, tdates)
    ax.set_ylabel(f"ours \u2212 {ref_name} ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: "$0" if x == 0 else f"${x:+,.0f}"))


def _plot_benchmark_panel(ax, merged: pd.DataFrame, tdates: list, spx, asset_avg) -> None:
    """plot portfolio vs S&P 500 and/or asset average, rebased to 100."""
    common = merged.index
    p_reb = merged["ours"] / merged["ours"].iloc[0] * 100
    if spx is not None:
        spx_a = spx.reindex(common, method="ffill").dropna()
        common_s = common.intersection(spx_a.index)
        b_reb = spx_a.loc[common_s] / spx_a.loc[common_s].iloc[0] * 100
        p_reb_s = merged["ours"].loc[common_s] / merged["ours"].loc[common_s].iloc[0] * 100
        ax.plot(common_s, b_reb, label="S&P 500", linewidth=1, color="#95A5A6", linestyle="--")
        ax.fill_between(common_s, p_reb_s, b_reb, where=p_reb_s >= b_reb, color="#27AE60", alpha=0.08)
        ax.fill_between(common_s, p_reb_s, b_reb, where=p_reb_s < b_reb, color="#E74C3C", alpha=0.08)
    if asset_avg is not None:
        aa = asset_avg.reindex(common, method="ffill").dropna()
        common_a = common.intersection(aa.index)
        aa_reb = aa.loc[common_a] / aa.loc[common_a].iloc[0] * 100
        ax.plot(common_a, aa_reb, label="asset avg (EW buy-hold)", linewidth=1, color="#E67E22", linestyle=":")
    ax.plot(common, p_reb, label="portfolio", linewidth=1.3, color=ENGINE_COLORS["ours"])
    _mark_trades(ax, tdates)
    ax.set_ylabel("Rebased (100)")
    ax.legend(loc="upper left")


def plot_equity_comparison(
    our_result, ours: pd.Series, refs: dict[str, pd.Series],
    merged: pd.DataFrame, results_dir: Path, title: str,
    spx: pd.Series | None = None, asset_avg: pd.Series | None = None,
) -> None:
    """render all comparison plots and save to results_dir/plots/."""
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    tdates = _trade_dates(our_result.trades)
    tset = set(tdates)
    dollar_fmt = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")

    with pub_style():
        n_refs = len(refs)
        has_bench = spx is not None or asset_avg is not None
        n_plots = 2 + n_refs + (1 if has_bench else 0)
        fig, axes = plt.subplots(
            n_plots, 1, figsize=(10, 2.4 + 2.2 * (n_plots - 1)),
            sharex=True, layout="constrained",
        )

        _plot_value_breakdown(axes[0], our_result)
        _mark_trades(axes[0], tdates)
        tv_trade = ours[ours.index.isin(tset)]
        axes[0].scatter(tv_trade.index, tv_trade.values, color="#A93226", marker="D", s=12, zorder=5, label="rebalance")
        axes[0].set_ylabel("Our engine ($)")
        axes[0].yaxis.set_major_formatter(dollar_fmt)
        axes[0].text(
            0.01, 0.96, title, transform=axes[0].transAxes, fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="#CCCCCC", linewidth=0.5),
        )
        axes[0].legend(loc="upper right", ncol=4)

        _plot_engine_overlay(axes[1], merged, tdates)

        for i, name in enumerate(refs):
            _plot_diff_panel(axes[2 + i], merged, name, tdates)

        if has_bench:
            _plot_benchmark_panel(axes[2 + n_refs], merged, tdates, spx, asset_avg)

        axes[-1].set_xlabel("Date")
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate(rotation=30, ha="right")

        fig.savefig(plots_dir / "equity-comparison.png")
        plt.close(fig)

    fig2 = our_result.plot()
    fig2.savefig(plots_dir / "engine-detail.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    logger.info("plots -> %s", plots_dir)
