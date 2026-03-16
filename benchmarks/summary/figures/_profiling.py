"""runtime profiling distribution figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from summary.figures._common import bid_color, pub_style, save, short_label
from utils.comparison import ENGINE_COLORS


def _draw_profiling_row(axes, df_prof, engines, labels, *,
                        sharex: bool, logscale: bool) -> None:
    """draw one row of box+strip panels across engines."""
    global_max = df_prof["wall_time_s"].max() * 1.1 if sharex else 0
    global_min = max(df_prof["wall_time_s"].min() * 0.5, 1e-4) if sharex else 1e-4
    for ax, eng in zip(axes, engines):
        edata = df_prof[df_prof["engine"] == eng]
        color = ENGINE_COLORS.get(eng, "#88CCEE")
        sns.boxplot(
            data=edata, y="label", x="wall_time_s", color=color,
            ax=ax, orient="h", linewidth=0.5, fliersize=0, width=0.55,
            order=labels, boxprops=dict(alpha=0.45),
            medianprops=dict(color="#333", linewidth=0.7),
        )
        sns.stripplot(
            data=edata, y="label", x="wall_time_s", color=color,
            ax=ax, orient="h", size=2.0 if logscale else 1.6,
            alpha=0.55, order=labels,
            jitter=0.12, legend=False,
        )
        ax.set_ylabel("")
        ax.grid(axis="x", linewidth=0.2, alpha=0.3)
        ax.tick_params(axis="x", labelsize=5)
        ax.tick_params(axis="y", labelsize=5.5)
        if logscale:
            ax.set_xscale("log")
            ax.set_xlim(global_min, global_max)
            ax.set_xlabel("log s", fontsize=5.5)
        elif sharex:
            ax.set_xlim(0, global_max)
            ax.set_xlabel("s", fontsize=5.5)
        else:
            ax.set_xlabel("s", fontsize=5.5)


def plot_profiling_distributions(df_prof: pd.DataFrame) -> None:
    """wall-time box+strip: 3 rows (shared-x, per-engine-x, log-scale) x engines."""
    engines = sorted(df_prof["engine"].unique())
    if not engines:
        return

    bids = sorted(df_prof["benchmark_id"].unique())
    labels = [short_label(b) for b in bids]
    bid_to_label = dict(zip(bids, labels))
    df_prof = df_prof.copy()
    df_prof["label"] = df_prof["benchmark_id"].map(bid_to_label)
    n_buckets = df_prof["bucket_id"].nunique()
    n_eng = len(engines)
    row_h = 0.20 * len(bids) + 0.35

    row_configs = [
        {"sharex": True, "logscale": False, "title": "shared scale"},
        {"sharex": False, "logscale": False, "title": "per-engine"},
        {"sharex": True, "logscale": True, "title": "log scale"},
    ]

    with pub_style():
        fig, all_axes = plt.subplots(
            3, n_eng,
            figsize=(2.5 * n_eng, row_h * 3),
            gridspec_kw={"hspace": 0.22, "wspace": 0.08},
        )
        if n_eng == 1:
            all_axes = all_axes[:, np.newaxis]

        for row_idx, cfg in enumerate(row_configs):
            row_axes = [all_axes[row_idx, ci] for ci in range(n_eng)]
            _draw_profiling_row(
                row_axes, df_prof, engines, labels,
                sharex=cfg["sharex"], logscale=cfg["logscale"],
            )
            for ci, eng in enumerate(engines):
                ax = row_axes[ci]
                if row_idx == 0:
                    ax.set_title(eng, fontsize=7, fontweight="bold",
                                 color=ENGINE_COLORS.get(eng, "#333"))
                if ci > 0:
                    ax.set_yticklabels([])
                if ci == 0:
                    ax.set_ylabel(cfg["title"], fontsize=6, fontweight="bold",
                                  color="#555")

        for ci, bid in enumerate(bids):
            ticks = all_axes[0, 0].get_yticklabels()
            if ci < len(ticks):
                for ri in range(3):
                    t = all_axes[ri, 0].get_yticklabels()
                    if ci < len(t):
                        t[ci].set_color(bid_color(bid))

        fig.subplots_adjust(top=0.94)
        fig.suptitle(
            f"engine wall time (n = {n_buckets} buckets)",
            fontsize=9, y=0.99,
        )
        save(fig, "profiling-distributions.png", tight=True, subdir="performance")
