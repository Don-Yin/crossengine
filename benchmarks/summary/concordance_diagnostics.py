"""pseudo-replication and floor decomposition diagnostics."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from summary.collect import collect_metrics, load_all_equity
from utils.wrappers import ACTIVE_ENGINES


def compute_bucket_exchangeability(
    n_bootstrap: int = 5000, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """cluster-bootstrap diagnostic for pseudo-replication; replaces lag-1 autocorrelation.

    resamples buckets with replacement, recomputes mean divergence per benchmark,
    and Spearman rho(cost, divergence). reports bootstrap CI. if buckets are
    dependent, bootstrap CI will be wider than naive.
    """
    from summary.figures._statistical_helpers import load_bucket_divergences
    from summary.figures._common import category as cat_fn

    bdf = load_bucket_divergences()
    if bdf is None:
        return pd.DataFrame(), pd.DataFrame(), {}
    df = collect_metrics()
    if df.empty or "total_commissions" not in df.columns or "total_slippage" not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), {}

    cost = (df["total_commissions"] + df["total_slippage"]).reindex(bdf["benchmark_id"].unique())
    bucket_max = bdf.groupby(["benchmark_id", "bucket_id"])["divergence"].max().reset_index()
    rng = np.random.default_rng(seed)

    buckets = sorted(bucket_max["bucket_id"].unique())
    n_buckets = len(buckets)

    bench_divs = _benchmark_divergence_arrays(bucket_max, buckets)
    obs_rho, obs_p = _spearman_cost_divergence(bucket_max, cost)
    rho_boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_buckets, size=n_buckets)
        resampled = _bootstrap_mean_divergence(bench_divs, cost, idx)
        rho_boot[i], _ = _spearman_cost_divergence(resampled, cost)
    ci_lo = float(np.percentile(rho_boot, 2.5))
    ci_hi = float(np.percentile(rho_boot, 97.5))

    detail_df = _exchangeability_detail(bucket_max, cost, cat_fn)
    summary = pd.DataFrame([{
        "diagnostic": "cluster_bootstrap_spearman_rho",
        "observed_rho": round(obs_rho, 4),
        "observed_p": round(obs_p, 4),
        "bootstrap_ci95_lower": round(ci_lo, 4),
        "bootstrap_ci95_upper": round(ci_hi, 4),
        "n_bootstrap": n_bootstrap,
        "interpretation": "robust" if ci_lo <= obs_rho <= ci_hi else "check_dependence",
    }])
    extra = {"observed_rho": obs_rho, "bootstrap_rhos": rho_boot}
    return detail_df, summary, extra


def _spearman_cost_divergence(bucket_agg: pd.DataFrame, cost: pd.Series) -> tuple[float, float]:
    """Spearman rho between benchmark-level cost and mean divergence."""
    mean_div = bucket_agg.groupby("benchmark_id")["divergence"].mean()
    common = mean_div.index.intersection(cost.dropna().index)
    if len(common) < 3:
        return 0.0, 1.0
    x, y = cost.loc[common].values, mean_div.loc[common].values
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def _benchmark_divergence_arrays(
    bucket_max: pd.DataFrame, buckets: list,
) -> dict[str, np.ndarray]:
    """per-benchmark array of divergences (one per bucket, ordered by bucket_id)."""
    out = {}
    for bid in bucket_max["benchmark_id"].unique():
        grp = bucket_max[bucket_max["benchmark_id"] == bid].sort_values("bucket_id")
        divs = grp["divergence"].values
        if len(divs) == len(buckets):
            out[bid] = divs
    return out


def _bootstrap_mean_divergence(
    bench_divs: dict[str, np.ndarray], cost: pd.Series, idx: np.ndarray,
) -> pd.DataFrame:
    """resample buckets with replacement; return mean divergence per benchmark."""
    rows = []
    for bid, divs in bench_divs.items():
        boot_mean = float(np.mean(divs[idx]))
        rows.append({"benchmark_id": bid, "bucket_id": "boot", "divergence": boot_mean})
    return pd.DataFrame(rows)


def _exchangeability_detail(bucket_max: pd.DataFrame, cost: pd.Series, cat_fn) -> pd.DataFrame:
    """per-benchmark mean divergence and cost for reporting."""
    mean_div = bucket_max.groupby("benchmark_id")["divergence"].mean()
    records = []
    for bid in mean_div.index:
        c = cost.get(bid, np.nan)
        if np.isfinite(c):
            records.append({
                "benchmark_id": bid, "category": cat_fn(bid),
                "mean_divergence": round(mean_div[bid], 6), "total_cost": round(c, 2),
            })
    return pd.DataFrame(records)


def compute_bucket_autocorrelation(results_root=None) -> tuple:
    """deprecated: lag-1 AC with arbitrary ordering is not a valid independence test.

    use compute_bucket_exchangeability() instead. kept for backward compatibility.
    """
    from summary.figures._statistical_helpers import load_bucket_divergences
    from summary.figures._common import category as cat_fn
    bdf = load_bucket_divergences()
    if bdf is None:
        return pd.DataFrame(), pd.DataFrame()
    autocorrs = []
    for (bid, pair), grp in bdf.groupby(["benchmark_id", "pair"]):
        vals = grp.sort_values("bucket_id")["divergence"].dropna().values
        if len(vals) < 10:
            continue
        centered = vals - vals.mean()
        denom = np.sum(centered ** 2)
        ac1 = float(np.sum(centered[:-1] * centered[1:]) / denom) if denom > 1e-15 else 0.0
        autocorrs.append({
            "benchmark_id": bid, "pair": pair, "category": cat_fn(bid),
            "n_buckets": len(vals), "lag1_autocorr": round(ac1, 6),
        })
    detail_df = pd.DataFrame(autocorrs)
    cat_summary = _summarize_autocorr_by_category(detail_df)
    return detail_df, cat_summary


def _summarize_autocorr_by_category(detail_df: pd.DataFrame) -> pd.DataFrame:
    """summarize lag-1 autocorrelation by category (deprecated)."""
    records = []
    for cat, grp in detail_df.groupby("category"):
        acs = grp["lag1_autocorr"].values
        n = len(acs)
        if n < 2:
            continue
        mean_ac = float(np.mean(acs))
        se_ac = float(np.std(acs, ddof=1) / np.sqrt(n))
        lo = round(mean_ac - 1.96 * se_ac, 4)
        hi = round(mean_ac + 1.96 * se_ac, 4)
        records.append({
            "category": cat, "n_pairs": n,
            "mean_lag1_ac": round(mean_ac, 4),
            "se_lag1_ac": round(se_ac, 4),
            "ci95_lower": lo, "ci95_upper": hi,
            "interpretation": "independent" if lo <= 0 <= hi else "mild clustering",
        })
    return pd.DataFrame(records)


def compute_floor_decomposition(results_root=None) -> pd.DataFrame:
    """decompose pairwise divergence into equity-recording convention floor and cost-model residual.

    the floor is estimated per engine-pair from BM01 (simplest nonzero-cost strategy).
    pairs crossing the {backtrader, cvxportfolio} vs {ours, bt, vectorbt} boundary
    show a constant ~0.18% offset from equity-recording timing differences.
    """
    equities = load_all_equity(results_root) if results_root else load_all_equity()
    if not equities:
        return pd.DataFrame()
    engines = list(ACTIVE_ENGINES)
    baseline_bid = "01-equal-weight"
    baseline_divs = _compute_pair_divs(equities.get(baseline_bid), engines)
    records = []
    for bid, edf in sorted(equities.items()):
        pair_divs = _compute_pair_divs(edf, engines)
        for pair_key, raw_div in pair_divs.items():
            floor_raw = baseline_divs.get(pair_key, 0.0) if bid != baseline_bid else 0.0
            floor_est = min(floor_raw, raw_div)
            records.append({
                "benchmark_id": bid, "pair": pair_key,
                "raw_div_pct": round(raw_div, 6),
                "floor_pct": round(floor_est, 6),
                "cost_residual_pct": round(raw_div - floor_est, 6),
            })
    return pd.DataFrame(records)


def _compute_pair_divs(edf, engines) -> dict[str, float]:
    """compute total return divergence for all engine pairs from one benchmark."""
    if edf is None:
        return {}
    result = {}
    for ea, eb in itertools.combinations(engines, 2):
        if ea not in edf.columns or eb not in edf.columns:
            continue
        a, b = edf[ea].dropna(), edf[eb].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        ret_a = (a.iloc[-1] / a.iloc[0] - 1) * 100
        ret_b = (b.iloc[-1] / b.iloc[0] - 1) * 100
        result[f"{ea}_vs_{eb}"] = abs(ret_a - ret_b)
    return result
