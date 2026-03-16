"""mechanism comparison and economic significance figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summary.figures._common import (
    CATEGORY_ORDER,
    CMAP_NATURE_REDS,
    add_n_label,
    bid_color,
    category,
    get_n_buckets,
    pair_color,
    pub_style,
    save,
    short_label,
)


def plot_mechanism_comparison(df: pd.DataFrame) -> None:
    """per-category mean divergence across engines -- heatmap with individual points."""
    div_cols = sorted(c for c in df.columns if c.startswith("div_") and c.endswith("_max_rel_pct"))
    if not div_cols:
        return
    sub = df[div_cols].dropna().copy()
    sub["cat"] = [category(b) for b in sub.index]
    eng_names = [c.replace("div_", "").replace("_max_rel_pct", "") for c in div_cols]
    eng_display = [e.replace("_vs_", " vs ") for e in eng_names]
    cat_order = [c for c in CATEGORY_ORDER if c in sub["cat"].values]

    mean_tbl, std_tbl = _build_cat_tables(sub, cat_order, div_cols, eng_names)
    n = get_n_buckets(df)
    long = _build_long(sub, div_cols, eng_names)

    with pub_style():
        fig, (ax_h, ax_s) = plt.subplots(
            1, 2, figsize=(10, 3.2),
            gridspec_kw={"width_ratios": [1.1, 1.4]},
            layout="constrained",
        )
        fig.get_layout_engine().set(wspace=0.04)

        _draw_mean_heatmap(ax_h, mean_tbl, std_tbl, cat_order, eng_names, eng_display)
        _draw_scatter_panel(ax_s, long, cat_order, eng_names)

        fig.suptitle("divergence by strategy category and engine", fontsize=9)
        if n:
            add_n_label(ax_s, n)
        save(fig, "mechanism-comparison.png", subdir="mechanism")


def _build_cat_tables(sub, cat_order, div_cols, eng_names):
    """compute per-category mean and std tables."""
    mean_tbl = pd.DataFrame(index=cat_order, columns=eng_names, dtype=float)
    std_tbl = pd.DataFrame(index=cat_order, columns=eng_names, dtype=float)
    for cat in cat_order:
        csub = sub[sub["cat"] == cat]
        for col, eng in zip(div_cols, eng_names):
            mean_tbl.loc[cat, eng] = csub[col].mean()
            std_tbl.loc[cat, eng] = csub[col].std()
    return mean_tbl, std_tbl


def _build_long(sub, div_cols, eng_names):
    """reshape wide divergence data to long form."""
    rows = []
    for _, r in sub.iterrows():
        for col, eng in zip(div_cols, eng_names):
            rows.append({"category": r["cat"], "engine": eng, "divergence": r[col]})
    return pd.DataFrame(rows)


def _draw_mean_heatmap(ax, mean_tbl, std_tbl, cat_order, eng_names, eng_display):
    """render the mean divergence heatmap with +/- std annotations."""
    vals = mean_tbl.values.astype(float)
    im = ax.imshow(vals, cmap=CMAP_NATURE_REDS, aspect="auto", vmin=0, vmax=max(vals.max(), 0.01))
    _annotate_heatmap_cells(ax, vals, std_tbl)
    ax.set_yticks(range(len(cat_order)))
    ax.set_yticklabels(cat_order, fontsize=7)
    ax.set_xticks(range(len(eng_names)))
    ax.set_xticklabels(eng_display, fontsize=5.5, rotation=35, ha="right")
    ax.set_title("mean divergence (%)", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)


def _annotate_heatmap_cells(ax, vals, std_tbl):
    """write value +/- std into each heatmap cell."""
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v, s = vals[i, j], std_tbl.iloc[i, j]
            tc = "white" if v > vals.max() * 0.55 else "black"
            txt = f"{v:.3f}" if v < 1 else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.5, color=tc, fontweight="bold")
            if np.isfinite(s):
                stxt = f"+/-{s:.3f}" if s < 1 else f"+/-{s:.2f}"
                ax.text(j, i + 0.28, stxt, ha="center", va="center", fontsize=3.5, color=tc, alpha=0.7)


def _draw_scatter_panel(ax, long, cat_order, eng_names):
    """render scatter panel of individual benchmark divergences."""
    for ci, cat in enumerate(cat_order):
        _draw_cat_engine_points(ax, long, cat, ci, eng_names)
    ax.set_yticks(range(len(cat_order)))
    ax.set_yticklabels(cat_order, fontsize=7)
    ax.set_xlabel("divergence (%)", fontsize=7)
    ax.set_title("individual benchmark divergences", fontsize=8)
    ax.grid(axis="x", linewidth=0.2, alpha=0.3)
    ax.tick_params(axis="x", labelsize=6)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=pair_color(e),
               markersize=4, label=e.replace("_vs_", " vs "))
               for e in eng_names]
    ax.legend(handles=handles, fontsize=5, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0, frameon=False)


def _draw_cat_engine_points(ax, long, cat, ci, eng_names):
    """draw jittered points for one category across all engines."""
    for eng in eng_names:
        color = pair_color(eng)
        cdata = long[(long["category"] == cat) & (long["engine"] == eng)]
        y_pos = np.full(len(cdata), ci)
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(cdata))
        ax.scatter(cdata["divergence"], y_pos + jitter, color=color, s=8, alpha=0.5, edgecolors="none")


def plot_economic_significance(df: pd.DataFrame, aum: float = 1_000_000) -> None:
    """dollar impact of engine divergence at reference AUM."""
    div_cols = sorted(c for c in df.columns if c.startswith("div_") and c.endswith("_max_rel_pct"))
    if not div_cols:
        return

    sub = df[div_cols].dropna().copy()
    sub_dollar = sub / 100 * aum
    order = sub_dollar.max(axis=1).sort_values(ascending=False).index
    sub_dollar = sub_dollar.loc[order]
    labels = [short_label(b) for b in order]
    n = get_n_buckets(df)
    n_eng = len(div_cols)
    bh = 0.75 / max(n_eng, 1)

    with pub_style():
        fig, ax = plt.subplots(figsize=(6, 0.30 * len(order) + 1.0), layout="constrained")
        y = np.arange(len(labels))
        _draw_dollar_bars(ax, y, sub_dollar, div_cols, bh, n_eng)
        ax.set_xscale("symlog", linthresh=1000)
        ax.set_yticks(y)
        ylbl = ax.set_yticklabels(labels, fontsize=7)
        for lbl, bid in zip(ylbl, order):
            lbl.set_color(bid_color(bid))
        ax.set_xlabel(f"dollar impact at ${aum / 1e6:.0f}M AUM (log scale)", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        ax.set_title("economic significance of engine divergence", fontsize=9)
        if n:
            add_n_label(ax, n)
        save(fig, "economic-significance.png", subdir="mechanism")


def _draw_dollar_bars(ax, y, sub_dollar, div_cols, bh, n_eng):
    """render grouped horizontal bars for dollar impact."""
    for j, col in enumerate(div_cols):
        eng = col.replace("div_", "").replace("_max_rel_pct", "")
        offset = (j - (n_eng - 1) / 2) * bh
        ax.barh(y + offset, sub_dollar[col], bh * 0.9,
                label=eng.replace("_vs_", " vs "),
                color=pair_color(eng),
                alpha=0.85, edgecolor="white", linewidth=0.5)
