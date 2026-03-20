"""Comparison reporting and publication-quality plotting."""

from __future__ import annotations

import itertools
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

from utils._comparison_plots import (
    ENGINE_COLORS,
    ENGINE_LINESTYLES,
    ENGINE_LINEWIDTHS,
    plot_equity_comparison,
    pub_style,
)

__all__ = [
    "ENGINE_COLORS",
    "ENGINE_LINESTYLES",
    "ENGINE_LINEWIDTHS",
    "pub_style",
    "write_comparison",
]


def _compute_pair_divergence(a: pd.Series, b: pd.Series) -> dict:
    """compute divergence stats between two equity curves."""
    diff = (a - b).abs()
    max_abs = float(diff.max())
    mid = (a + b) / 2
    max_rel = float((diff / mid).max() * 100)
    a_ret = float((a.iloc[-1] / a.iloc[0] - 1) * 100)
    b_ret = float((b.iloc[-1] / b.iloc[0] - 1) * 100)
    return {
        "max_abs_diff": round(max_abs, 4),
        "max_rel_diff_pct": round(max_rel, 4),
        "a_return_pct": round(a_ret, 4),
        "b_return_pct": round(b_ret, 4),
    }


def _sanitize(obj):
    """replace inf/-inf/nan with None for JSON serialization."""
    if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _append_divergence_lines(lines: list, divergence: dict) -> None:
    """append human-readable pairwise divergence lines to report."""
    for pair_key, d in divergence.items():
        lines.append(
            f"  {pair_key:<30}  max |diff| = ${d['max_abs_diff']:>10.2f}"
            f"   max rel = {d['max_rel_diff_pct']:.4f}%"
        )


def _add_benchmark_section(
    lines: list, ours: pd.Series, bench: pd.Series | None, heading: str,
) -> dict | None:
    """compute benchmark metrics and append report lines if available."""
    if bench is None:
        return None
    from crossengine.metrics import compute_benchmark_metrics

    bm = compute_benchmark_metrics(ours, bench)
    if not bm:
        return None
    lines += [
        "", f"--- {heading} ---", "",
        f"  Excess return:      {bm['excess_return_pct']:>10.4f}%",
        f"  Beta:               {bm['beta']:>10.4f}",
        f"  Alpha (ann.):       {bm['alpha_ann_pct']:>10.4f}%",
        f"  Information ratio:  {bm['information_ratio']:>10.4f}",
        f"  Tracking error:     {bm['tracking_error_pct']:>10.4f}%",
        f"  Up capture:         {bm['up_capture']:>10.4f}",
        f"  Down capture:       {bm['down_capture']:>10.4f}",
    ]
    return bm


def write_comparison(
    our_result,
    refs: dict[str, pd.Series],
    results_dir: Path,
    title: str,
    note: str | None = None,
    spx: pd.Series | None = None,
    asset_avg: pd.Series | None = None,
) -> None:
    """write report + plots comparing our engine against references."""
    results_dir.mkdir(parents=True, exist_ok=True)

    ours = our_result.portfolio_value
    cols = {"ours": ours}
    cols.update(refs)
    merged = pd.DataFrame(cols).dropna()
    merged.to_csv(results_dir / "equity.csv")

    all_engines = {"ours": ours, **refs}

    hdr = f"{'Metric':<25}"
    for name in ["ours"] + list(refs.keys()):
        hdr += f" {name:>14}"
    sep = "-" * len(hdr)

    row_start = f"{'Start value':<25} {merged['ours'].iloc[0]:>14.2f}"
    row_end = f"{'End value':<25} {merged['ours'].iloc[-1]:>14.2f}"
    our_ret = (merged["ours"].iloc[-1] / merged["ours"].iloc[0] - 1) * 100
    row_ret = f"{'Total return %':<25} {our_ret:>14.4f}"

    for name in refs:
        row_start += f" {merged[name].iloc[0]:>14.2f}"
        row_end += f" {merged[name].iloc[-1]:>14.2f}"
        r = (merged[name].iloc[-1] / merged[name].iloc[0] - 1) * 100
        row_ret += f" {r:>14.4f}"

    lines = [
        title.upper(), "=" * 50, "",
        "--- engine metrics ---", "",
        our_result.report, "", "",
        "--- comparison vs references ---", "",
        hdr, sep, row_start, row_end, row_ret, "",
    ]

    divergence = {}
    for a_name, b_name in itertools.combinations(sorted(all_engines.keys()), 2):
        pair_key = f"{a_name}_vs_{b_name}"
        divergence[pair_key] = _compute_pair_divergence(merged[a_name], merged[b_name])

    n = len(all_engines)
    k = len(divergence)
    lines.append(f"--- pairwise divergence (all C({n},2) = {k} pairs) ---")
    lines.append("")
    _append_divergence_lines(lines, divergence)

    lines += [
        "", "--- sources of divergence ---", "",
        "  all engines receive the same total cost rate (commission + slippage).",
        "  our engine splits this into commission + slippage (price impact);",
        "  references receive the full amount as a proportional fee.",
        "  remaining divergence comes from:",
        "  - cost integration: engines differ in when fees are deducted relative",
        "    to target share calculation, causing small compounding differences",
        "  - execution order: sell-before-buy vs concurrent vs buy-first",
        "  - cash settlement: intra-bar availability of sell proceeds for same-bar",
        "    buys varies by engine",
    ]

    benchmark_metrics = _add_benchmark_section(lines, ours, spx, "vs S&P 500")
    asset_avg_metrics = _add_benchmark_section(lines, ours, asset_avg, "vs condition asset average (EW buy-hold)")

    if note:
        lines += ["", f"Note: {note}"]

    report_text = "\n".join(lines)
    (results_dir / "report.txt").write_text(report_text)

    m = our_result.metrics
    max_div = max((d["max_rel_diff_pct"] for d in divergence.values()), default=0.0)
    logger.info("%s  ret=%.1f%%  sharpe=%.2f  maxDD=%.1f%%  max_div=%.2f%%",
                title, m["total_return_pct"], m["sharpe_ratio"],
                m["max_drawdown_pct"], max_div)
    logger.info("report  -> %s", results_dir / "report.txt")

    from utils.data import RESULTS_ROOT

    resolved = results_dir.resolve()
    root_resolved = RESULTS_ROOT.resolve()
    benchmark_id = (
        str(resolved.relative_to(root_resolved))
        if resolved.is_relative_to(root_resolved)
        else results_dir.name
    )

    payload = {
        "schema_version": 2,
        "benchmark_id": benchmark_id,
        "title": title,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "engine_metrics": our_result.metrics,
        "benchmark_metrics": benchmark_metrics,
        "asset_avg_metrics": asset_avg_metrics,
        "divergence": divergence,
    }
    (results_dir / "metrics.json").write_text(json.dumps(_sanitize(payload), indent=2) + "\n")
    logger.info("metrics -> %s", results_dir / "metrics.json")

    plot_equity_comparison(our_result, ours, refs, merged, results_dir, title, spx, asset_avg)
