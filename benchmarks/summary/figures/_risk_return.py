"""risk-return faceted KDE figure: CAGR vs max drawdown per benchmark."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summary.figures._common import bid_color, pub_style, save, short_label


def plot_risk_return(df: pd.DataFrame) -> None:
    """faceted KDE of CAGR vs max drawdown, one subplot per benchmark."""
    from summary.bucket_data import collect_bucket_metrics

    points = _build_risk_return(collect_bucket_metrics(sections=("engine",)))
    if len(points) < 10:
        return
    bids = sorted(points["benchmark_id"].unique())
    xr = np.percentile(points["dd_abs"].dropna(), [2, 98])
    yr = np.percentile(points["cagr"].dropna(), [2, 98])
    xp, yp = (xr[1] - xr[0]) * 0.12, (yr[1] - yr[0]) * 0.12
    ncols, nrows = 5, -(-len(bids) // 5)
    with pub_style():
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.2 * nrows), sharex=True, sharey=True, layout="constrained")
        for i, bid in enumerate(bids):
            _draw_bench_kde(axes.flat[i], points[points["benchmark_id"] == bid], bid)
            axes.flat[i].set_xlim(xr[0] - xp, xr[1] + xp)
            axes.flat[i].set_ylim(yr[0] - yp, yr[1] + yp)
        for j in range(len(bids), axes.size):
            axes.flat[j].set_visible(False)
        fig.supxlabel("max drawdown (%)", fontsize=8)
        fig.supylabel("CAGR (%)", fontsize=8)
        fig.suptitle(f"risk-return (n = {points['bucket_id'].nunique()} buckets)", fontsize=9)
        save(fig, "risk-return.png", subdir="performance")


def _build_risk_return(long: pd.DataFrame) -> pd.DataFrame:
    """pivot bucket metrics to per-point cagr and drawdown."""
    cols = ["benchmark_id", "bucket_id", "value"]
    cagr = long.loc[long["metric"] == "cagr_pct", cols].rename(columns={"value": "cagr"})
    dd = long.loc[long["metric"] == "max_drawdown_pct", cols].rename(columns={"value": "dd"})
    m = cagr.merge(dd, on=["benchmark_id", "bucket_id"])
    m["dd_abs"] = m["dd"].abs()
    return m


def _draw_bench_kde(ax, grp: pd.DataFrame, bid: str) -> None:
    """scatter + KDE contours for one benchmark subplot."""
    import seaborn as sns

    c = bid_color(bid)
    ax.scatter(grp["dd_abs"], grp["cagr"], color=c, s=10, alpha=0.5, edgecolors="none", zorder=3)
    if len(grp) >= 5:
        sns.kdeplot(data=grp, x="dd_abs", y="cagr", ax=ax, color=c, levels=3, fill=True, alpha=0.15, zorder=2)
        sns.kdeplot(data=grp, x="dd_abs", y="cagr", ax=ax, color=c, levels=3, linewidths=0.6, fill=False, zorder=2)
    ax.set_title(short_label(bid), fontsize=6, fontweight="bold", color=c)
    ax.set(xlabel="", ylabel="")
