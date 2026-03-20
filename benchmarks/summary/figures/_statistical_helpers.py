"""computation helpers for statistical robustness analyses."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from summary.concordance import collect_bucket_engine_stats
from summary.figures._common import category


def load_bucket_divergences(metric: str = "total_return_pct") -> pd.DataFrame | None:
    """load per-bucket pairwise engine divergence from bucket stats."""
    raw = collect_bucket_engine_stats()
    if raw.empty:
        print("  warn: no bucket data available for statistical tests")
        return None
    sub = raw[raw["metric"] == metric]
    wide = sub.pivot_table(
        index=["benchmark_id", "bucket_id"], columns="engine", values="value",
    )
    engines = sorted(wide.columns)
    frames = [
        _pair_divergence(wide, ea, eb)
        for ea, eb in itertools.combinations(engines, 2)
    ]
    if not frames:
        return None
    result = pd.concat(frames, ignore_index=True)
    if result.empty:
        return None
    result["category"] = result["benchmark_id"].map(category)
    return result


def _pair_divergence(wide: pd.DataFrame, ea: str, eb: str) -> pd.DataFrame:
    """compute signed divergence between two engines across all buckets."""
    diff = (wide[ea] - wide[eb]).dropna().reset_index()
    diff.columns = ["benchmark_id", "bucket_id", "divergence"]
    diff["pair"] = f"{ea}_vs_{eb}"
    return diff


def tost_pvalue(values: np.ndarray, margin: float) -> float:
    """two one-sided t-tests p-value for equivalence within +/- margin."""
    n = len(values)
    if n < 3:
        return 1.0
    mean, se = np.mean(values), np.std(values, ddof=1) / np.sqrt(n)
    if se < 1e-15:
        return 0.0 if abs(mean) < margin else 1.0
    p_upper = stats.t.cdf((mean - margin) / se, df=n - 1)
    p_lower = stats.t.sf((mean + margin) / se, df=n - 1)
    return float(max(p_upper, p_lower))


def tost_power(n: int, std: float, margin: float, alpha: float = 0.05) -> float:
    """statistical power of tost assuming true mean = 0."""
    if n < 3:
        return 0.0
    if std < 1e-12:
        return 1.0
    delta = margin / (std / np.sqrt(n))
    t_crit = stats.t.ppf(alpha, df=n - 1)
    return float(stats.nct.cdf(t_crit, df=n - 1, nc=-delta))


def permutation_pvalue(
    values: np.ndarray, n_perm: int = 10_000, seed: int = 42,
) -> tuple[float, np.ndarray, float]:
    """sign-flip permutation test for mean divergence != 0."""
    observed = abs(np.mean(values))
    if observed < 1e-6:
        return observed, np.array([]), 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, len(values)))
    perm_means = np.abs((values[None, :] * signs).mean(axis=1))
    p = float((np.sum(perm_means >= observed) + 1) / (n_perm + 1))
    return observed, perm_means, p


