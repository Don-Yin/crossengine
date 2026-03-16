"""shared constants, helpers, and category system for summary figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap

from summary.collect import SUMMARY_DIR
from utils.comparison import pub_style  # noqa: F401 -- re-exported

CMAP_NATURE_REDS = LinearSegmentedColormap.from_list(
    "nature_reds", ["#FFFFFF", "#FADBD8", "#E64B35", "#91372B"],
)
CMAP_NATURE_BLUES = LinearSegmentedColormap.from_list(
    "nature_blues", ["#FFFFFF", "#D6EAF8", "#4DBBD5", "#3C5488"],
)

FIGURES_DIR = SUMMARY_DIR / "figures"

CATEGORY_MAP = {
    "01-equal-weight": "simple",
    "02-stay-drift": "simple",
    "03-rotation": "rotation",
    "04-rotation-with-cost": "rotation",
    "10-cash-starved": "rotation",
    "11-concentrated-cascade": "rotation",
    "05-sma-momentum": "signal",
    "06-inverse-vol": "simple",
    "07-cross-momentum": "signal",
    "09-daily-binary-switch": "ablation",
    "12-daily-equal-weight": "simple",
}
_ML_PREFIX = "08-ml-signal"

CATEGORY_ORDER = ["simple", "signal", "rotation", "ml", "ablation"]
CATEGORY_TITLES = {
    "simple": "simple / baseline",
    "signal": "signal-based",
    "rotation": "rotation / allocation",
    "ml": "machine learning",
    "ablation": "ablation / stress-test",
}
NPG_PALETTE = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]

CATEGORY_COLORS = {
    "simple": "#00A087", "rotation": "#E64B35", "signal": "#3C5488",
    "ml": "#8491B4", "ablation": "#7E6148",
}

BENCHMARK_COLORS = {
    "01-equal-weight": "#00A087", "02-stay-drift": "#006D5B",
    "03-rotation": "#E64B35", "04-rotation-with-cost": "#B83226",
    "10-cash-starved": "#F39B7F", "11-concentrated-cascade": "#91372B",
    "05-sma-momentum": "#3C5488", "06-inverse-vol": "#91D1C2",
    "07-cross-momentum": "#2A3D66",
    "08-ml-signal/gbr": "#8491B4", "08-ml-signal/enet": "#91D1C2",
    "08-ml-signal/rf": "#6B7A9E", "08-ml-signal/mlp": "#5A6888",
    "09-daily-binary-switch": "#7E6148", "12-daily-equal-weight": "#3CB371",
}

BENCHMARK_STYLES = {
    "01-equal-weight": "-", "02-stay-drift": "--",
    "03-rotation": "-", "04-rotation-with-cost": "--",
    "10-cash-starved": "-.", "11-concentrated-cascade": ":",
    "05-sma-momentum": "-", "06-inverse-vol": "--",
    "07-cross-momentum": "-.",
    "08-ml-signal/gbr": "-", "08-ml-signal/enet": "--",
    "08-ml-signal/rf": "-.", "08-ml-signal/mlp": ":",
    "09-daily-binary-switch": "-", "12-daily-equal-weight": "--",
}


_n_map: dict[str, int] = {}


def register_n_buckets(df) -> None:
    """cache bucket counts so short_label can append (n=xx) automatically."""
    _n_map.clear()
    if hasattr(df, "columns") and "n_buckets" in df.columns:
        for bid in df.index:
            _n_map[bid] = int(df.loc[bid, "n_buckets"])


def category(bid: str) -> str:
    """resolve benchmark id to its category name."""
    if bid.startswith(_ML_PREFIX):
        return "ml"
    return CATEGORY_MAP.get(bid, "simple")


def short_label(bid: str) -> str:
    """shorten benchmark id for axis labels, with (n=xx) when registered."""
    lbl = bid.replace("08-ml-signal/", "ml-")
    n = _n_map.get(bid)
    if n is not None:
        lbl += f" (n={n})"
    return lbl


def bid_color(bid: str) -> str:
    """per-benchmark unique color."""
    return BENCHMARK_COLORS.get(bid, "#333333")


def bid_style(bid: str) -> str:
    """per-benchmark line style."""
    return BENCHMARK_STYLES.get(bid, "-")


def group_by_category(bids: list[str]) -> dict[str, list[str]]:
    """group benchmark ids by category, preserving CATEGORY_ORDER."""
    groups: dict[str, list[str]] = {c: [] for c in CATEGORY_ORDER}
    for bid in sorted(bids):
        cat = category(bid)
        groups.setdefault(cat, []).append(bid)
    return {c: v for c, v in groups.items() if v}


def save(fig, name: str, *, tight: bool = False, subdir: str = "") -> None:
    """save figure into an optional subdirectory and close it."""
    dest = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / name
    kw = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(p, **kw)
    plt.close(fig)
    print(f"  plot -> {p}")


def add_category_legend(ax) -> None:
    """add a small legend mapping category colors to names."""
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=CATEGORY_COLORS[c], label=CATEGORY_TITLES[c])
        for c in CATEGORY_ORDER if c in CATEGORY_COLORS
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6, framealpha=0.9)


def annotate_cells(ax, values, normed, fmt_fn, threshold: float = 0.6) -> None:
    """write text annotations into every cell of a heatmap-style plot."""
    for i, j in np.ndindex(values.shape):
        txt = fmt_fn(values[i, j], normed[i, j])
        text_color = "white" if normed[i, j] > threshold else "black"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=5.5, color=text_color)


def add_n_label(ax_or_fig, n: int, loc: str = "figure") -> None:
    """annotate the bucket count on a figure or axis."""
    txt = f"n = {n} buckets"
    if loc == "figure" and hasattr(ax_or_fig, "text"):
        ax_or_fig.text(0.99, 0.01, txt, transform=ax_or_fig.transAxes,
                       fontsize=6, color="#888", ha="right", va="bottom")
    else:
        ax_or_fig.annotate(txt, xy=(1, 0), xycoords="axes fraction",
                           fontsize=6, color="#888", ha="right", va="bottom")


def get_pop_std(df, metric: str):
    """get the population std column for a metric, returning 0 if missing."""
    col = f"pop_std_{metric}"
    if col in df.columns:
        return df[col].fillna(0)
    return 0


def get_n_buckets(df) -> int:
    """extract the common n_buckets from the dataframe."""
    if "n_buckets" in df.columns:
        return int(df["n_buckets"].max())
    return 0


def labels_and_colors(index) -> tuple[list[str], list[str]]:
    """build label text and color lists from a benchmark index."""
    return ([short_label(bid) for bid in index],
            [bid_color(bid) for bid in index])


def apply_label_colors(label_objs, colors: list[str]) -> None:
    """set color on each tick label object."""
    for lbl, col in zip(label_objs, colors):
        lbl.set_color(col)


def normalize_cols(sub, lower_better: set[str] | None = None):
    """min-max normalize dataframe columns, flipping lower-is-better metrics."""
    lower_better = lower_better or set()
    normed = sub.copy()
    for c in lower_better:
        if c in normed.columns:
            normed[c] = -normed[c]
    for c in normed.columns:
        rng = normed[c].max() - normed[c].min()
        if rng > 1e-10:
            normed[c] = (normed[c] - normed[c].min()) / rng
    return normed


def make_col_labels(present: list[str], lower_better: set[str] | None = None) -> list[str]:
    """human-readable column labels with | markers for inverted metrics."""
    lower_better = lower_better or set()
    out = []
    for c in present:
        lbl = c.replace("_pct", " (%)").replace("_", " ")
        if c in lower_better:
            lbl = "|" + lbl + "|"
        out.append(lbl)
    return out


def pair_label(col: str) -> str:
    """extract human-readable pair name from a divergence column."""
    return col.replace("div_", "").replace("_max_rel_pct", "").replace("_vs_", " vs ")


_PAIR_COLOR_CACHE: dict[str, str] = {}
_PAIR_COUNTER = 0


def pair_color(pair_key: str) -> str:
    """deterministic color for an engine-pair key, maximising visual contrast."""
    global _PAIR_COUNTER
    if pair_key not in _PAIR_COLOR_CACHE:
        _PAIR_COLOR_CACHE[pair_key] = NPG_PALETTE[_PAIR_COUNTER % len(NPG_PALETTE)]
        _PAIR_COUNTER += 1
    return _PAIR_COLOR_CACHE[pair_key]
