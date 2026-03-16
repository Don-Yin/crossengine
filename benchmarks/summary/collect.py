"""Auto-discover benchmark results and build a summary DataFrame.

The only contract: each benchmark writes a ``metrics.json`` alongside its
``equity.csv``.  This module discovers them via ``rglob``, so adding a new
benchmark or sub-benchmark requires zero changes here.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

from utils.data import RESULTS_ROOT
from utils.wrappers import ACTIVE_ENGINES

SUMMARY_DIR = RESULTS_ROOT / "summary"


def _prefix_dict(d: dict, prefix: str) -> dict:
    """prefix all keys of a flat dict."""
    return {f"{prefix}{k}": v for k, v in d.items()} if d else {}


def _flatten_divergence(div: dict) -> dict:
    """extract per-pair divergence fields into flat keys."""
    out = {}
    for pair_name, metrics in div.items():
        if not isinstance(metrics, dict):
            continue
        out[f"div_{pair_name}_max_abs"] = metrics.get("max_abs_diff")
        out[f"div_{pair_name}_max_rel_pct"] = metrics.get("max_rel_diff_pct")
        out[f"div_{pair_name}_a_return_pct"] = metrics.get("a_return_pct")
        out[f"div_{pair_name}_b_return_pct"] = metrics.get("b_return_pct")
    return out


def _flatten_population(pop: dict) -> dict:
    """extract population std/min/max per metric into flat pop_std_*, pop_min_*, pop_max_* keys."""
    out = {}
    for metric, stats in pop.items():
        if not isinstance(stats, dict):
            continue
        for stat_name in ("std", "min", "max"):
            val = stats.get(stat_name)
            if val is not None:
                out[f"pop_{stat_name}_{metric}"] = val
    return out


def discover_benchmarks(results_root: Path = RESULTS_ROOT) -> list[dict]:
    """recursively find all aggregate metrics.json, skipping per-bucket sub-results."""
    paths = sorted(results_root.rglob("metrics.json"))
    return [json.loads(p.read_text()) for p in paths if "buckets" not in p.relative_to(results_root).parts]


def collect_metrics(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """load all benchmarks into a flat dataframe, one row per benchmark."""
    records = discover_benchmarks(results_root)
    rows = []
    for rec in records:
        row = {
            "benchmark_id": rec.get("benchmark_id"),
            "title": rec.get("title"),
        }
        if rec.get("engine_metrics"):
            row.update(rec["engine_metrics"])
        if rec.get("benchmark_metrics"):
            row.update(rec["benchmark_metrics"])
        row.update(_prefix_dict(rec.get("asset_avg_metrics", {}), "aa_"))
        row.update(_flatten_divergence(rec.get("divergence", {})))
        row.update(_flatten_population(rec.get("population", {})))
        row["n_buckets"] = rec.get("n_buckets", 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("benchmark_id")
    return df


def load_equity(benchmark_id: str, results_root: Path = RESULTS_ROOT) -> pd.DataFrame | None:
    """load equity.csv for a specific benchmark, or None if missing."""
    p = results_root / benchmark_id / "equity.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0, parse_dates=True)


def load_all_equity(
    results_root: Path = RESULTS_ROOT,
    records: list[dict] | None = None,
) -> dict[str, pd.DataFrame]:
    """load equity curves for every discovered benchmark."""
    if records is None:
        records = discover_benchmarks(results_root)
    out = {}
    for rec in records:
        bid = rec.get("benchmark_id")
        if bid is None:
            continue
        df = load_equity(bid, results_root)
        if df is not None:
            out[bid] = df
    return out


def collect_return_correlations(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """pairwise correlation of daily returns (our engine) across all benchmarks."""
    equities = load_all_equity(results_root)
    series = {}
    for bid, df in equities.items():
        if "ours" in df.columns:
            series[bid] = df["ours"].pct_change().dropna()
    if not series:
        return pd.DataFrame()
    combined = pd.DataFrame(series)
    return combined.corr()


def _collect_pairs_for_benchmark(series: dict, bid: str, df: pd.DataFrame) -> None:
    """collect pairwise first-differenced divergence series for one benchmark."""
    engine_cols = sorted(c for c in df.columns if c in set(ACTIVE_ENGINES))
    for a, b in itertools.combinations(engine_cols, 2):
        diff = (df[a] - df[b]).diff().dropna()
        series[f"{bid}_{a}_vs_{b}"] = diff


def collect_divergence_correlations(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """pairwise correlation of first-differenced divergence across all engine pairs."""
    equities = load_all_equity(results_root)
    series: dict[str, pd.Series] = {}
    for bid, df in equities.items():
        _collect_pairs_for_benchmark(series, bid, df)
    if not series:
        return pd.DataFrame()
    combined = pd.DataFrame(series)
    return combined.corr()


TABLES_RAW_DIR = SUMMARY_DIR / "tables" / "raw"
TABLES_SPX_DIR = SUMMARY_DIR / "tables" / "spx-controlled"
TABLES_AA_DIR = SUMMARY_DIR / "tables" / "asset-controlled"


def _write_table(target_dir: Path, cols: list[str], rename_fn=None, results_root: Path = RESULTS_ROOT) -> Path:
    """shared helper: select columns from metrics, rename, and write csv."""
    target_dir.mkdir(parents=True, exist_ok=True)
    df = collect_metrics(results_root)
    p = target_dir / "metrics.csv"
    if df.empty:
        return p
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return p
    out = df[keep].copy()
    if rename_fn:
        out.columns = [rename_fn(c) for c in out.columns]
    else:
        out.columns = [c.replace("_pct", " (%)").replace("_", " ") for c in out.columns]
    out.to_csv(p)
    print(f"  table -> {p}")
    return p


def write_summary_csv(results_root: Path = RESULTS_ROOT) -> Path:
    """write full (uncontrolled) metrics to tables/raw/metrics.csv."""
    df = collect_metrics(results_root)
    div_cols = sorted(c for c in df.columns if c.startswith("div_"))
    return _write_table(
        TABLES_RAW_DIR,
        [
            "title",
            "cagr_pct",
            "ann_volatility_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "omega_ratio",
            "max_drawdown_pct",
            "var_95_pct",
            "cvar_95_pct",
            "skewness",
            "kurtosis",
            "win_rate_pct",
            "num_trades",
            "total_commissions",
            "total_slippage",
        ]
        + div_cols,
        results_root=results_root,
    )


def write_benchmark_relative_csv(results_root: Path = RESULTS_ROOT) -> Path:
    """write SPX-controlled metrics to tables/spx-controlled/metrics.csv."""
    return _write_table(
        TABLES_SPX_DIR,
        [
            "title",
            "cagr_pct",
            "ann_volatility_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown_pct",
            "win_rate_pct",
            "excess_return_pct",
            "alpha_ann_pct",
            "beta",
            "information_ratio",
            "tracking_error_pct",
            "up_capture",
            "down_capture",
        ],
        results_root=results_root,
    )


def write_asset_avg_csv(results_root: Path = RESULTS_ROOT) -> Path:
    """write asset-average-controlled metrics to tables/asset-controlled/metrics.csv."""
    return _write_table(
        TABLES_AA_DIR,
        [
            "title",
            "cagr_pct",
            "sharpe_ratio",
            "aa_excess_return_pct",
            "aa_alpha_ann_pct",
            "aa_beta",
            "aa_information_ratio",
            "aa_tracking_error_pct",
            "aa_up_capture",
            "aa_down_capture",
        ],
        rename_fn=lambda c: c.replace("aa_", "").replace("_pct", " (%)").replace("_", " "),
        results_root=results_root,
    )
