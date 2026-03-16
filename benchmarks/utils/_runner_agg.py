"""aggregation helpers for multi-bucket benchmark results."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

_INTEGER_FIELDS = {"num_trades", "trading_days"}


def collect_numeric(records: list[dict], section: str) -> dict[str, list[float]]:
    """extract all finite numeric values per metric key from a flat json section."""
    collected: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        sec = rec.get(section, {})
        if isinstance(sec, dict):
            _append_finite(collected, sec)
    return collected


def _append_finite(target: dict[str, list[float]], source: dict) -> None:
    """append all finite numeric values from source into target lists."""
    for k, v in source.items():
        if isinstance(v, (int, float)) and np.isfinite(v):
            target[k].append(v)


def mean_dict(collected: dict[str, list[float]]) -> dict:
    """compute mean for each metric, preserving integer fields."""
    out = {}
    for k, vals in collected.items():
        if not vals:
            continue
        mean = float(np.mean(vals))
        out[k] = int(round(mean)) if k in _INTEGER_FIELDS else round(mean, 4)
    return out


def stats_dict(collected: dict[str, list[float]]) -> dict:
    """compute population stats (mean, std, min, max, median, n) per metric."""
    out = {}
    for k, vals in collected.items():
        arr = np.array(vals)
        out[k] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else 0.0,
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "median": round(float(np.median(arr)), 4),
            "n": len(arr),
        }
    return out


def agg_divergence(records: list[dict]) -> dict:
    """mean of divergence section {pair_key: {metric: value}} across buckets."""
    groups: dict[str, dict[str, list[float]]] = {}
    pairs = ((pk, m) for rec in records for pk, m in rec.get("divergence", {}).items() if isinstance(m, dict))
    for pair_key, metrics in pairs:
        groups.setdefault(pair_key, defaultdict(list))
        _append_finite(groups[pair_key], metrics)
    return {g: _mean_from_lists(m) for g, m in groups.items()}


def _mean_from_lists(m: dict[str, list[float]]) -> dict[str, float]:
    """compute mean for each key's value list, dropping empties."""
    return {k: round(float(np.mean(vals)), 4) for k, vals in m.items() if vals}
