"""bucket distribution plots for multi-bucket runner."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

_PUB_RC: dict = {
    "font.family": "sans-serif", "font.size": 7, "pdf.fonttype": 42,
    "axes.linewidth": 0.5, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 5, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
}

_TARGETS = [
    ("total_return_pct", "total return (%)"), ("cagr_pct", "CAGR (%)"),
    ("sharpe_ratio", "sharpe ratio"), ("max_drawdown_pct", "max drawdown (%)"),
]


def _collect_for_plot(records: list[dict]) -> dict[str, list[float]]:
    """extract finite numeric values from engine_metrics for plotting."""
    collected: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        sec = rec.get("engine_metrics", {})
        if not isinstance(sec, dict):
            continue
        for k, v in sec.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                collected[k].append(float(v))
    return collected


def _draw_histogram(ax, vals: list[float], label: str, is_first: bool) -> None:
    """render a single histogram panel for one metric."""
    ax.hist(vals, bins=min(12, max(5, len(vals) // 3)),
            color="#0072B2", edgecolor="white", linewidth=0.3, alpha=0.85)
    m = np.mean(vals)
    ax.axvline(m, color="#D55E00", lw=0.8, ls="--", label=f"mean = {m:.2f}")
    ax.set_xlabel(label)
    ax.legend(fontsize=5)
    if is_first:
        ax.set_ylabel("count")


def plot_distributions(plots_dir: Path, records: list[dict]) -> None:
    """histogram of key metrics across buckets."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    values = _collect_for_plot(records)
    active = [(k, label) for k, label in _TARGETS if values.get(k)]
    if not active:
        return
    with plt.rc_context(_PUB_RC):
        n = len(active)
        fig, axes = plt.subplots(1, n, figsize=(7.2, 2.2), layout="constrained")
        if n == 1:
            axes = [axes]
        for ax, (key, label) in zip(axes, active):
            _draw_histogram(ax, values[key], label, is_first=(ax is axes[0]))
        fig.suptitle(f"distribution across {len(records)} buckets", fontsize=8)
        fig.savefig(plots_dir / "bucket-distributions.png")
        plt.close(fig)
    logger.info("bucket-distributions.png")
