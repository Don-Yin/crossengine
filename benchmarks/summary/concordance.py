"""engine concordance computation -- quantifies implementation risk across engines."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from summary.collect import discover_benchmarks, load_all_equity
from utils.data import RESULTS_ROOT
from utils.wrappers import ACTIVE_ENGINES


def compute_engine_concordance(results_root=None) -> pd.DataFrame:
    """compute key metrics from all five engines' equity curves.

    returns a dataframe indexed by benchmark_id with columns for
    per-engine metrics, engine sensitivity (ES) scores, and
    per-benchmark bucket-level CCC (computed across 30 buckets).
    """
    equities = load_all_equity(results_root) if results_root else load_all_equity()
    if not equities:
        return pd.DataFrame()

    engines = list(ACTIVE_ENGINES)
    metrics = ["total_return_pct", "cagr_pct", "ann_vol_pct", "sharpe", "max_dd_pct"]

    rows = [_engine_row(bid, edf, engines) for bid, edf in sorted(equities.items())]
    df = pd.DataFrame(rows).set_index("benchmark_id")
    bucket_cccs = _compute_bucket_cccs(metrics)
    es_rows = [_es_row(bid, df, engines, metrics, bucket_cccs) for bid in df.index]
    conc = pd.DataFrame(es_rows).set_index("benchmark_id")
    _add_daf(conc, metrics)
    return conc


def _engine_row(bid: str, edf: pd.DataFrame, engines: list[str]) -> dict:
    """compute per-engine financial metrics for one benchmark."""
    row: dict = {"benchmark_id": bid}
    for eng in engines:
        if eng not in edf.columns:
            continue
        stats = _equity_stats(edf[eng].dropna())
        if stats is None:
            continue
        row[f"total_return_pct_{eng}"] = stats["total_return_pct"]
        row[f"cagr_pct_{eng}"] = stats["cagr_pct"]
        row[f"ann_vol_pct_{eng}"] = stats["ann_vol_pct"]
        row[f"sharpe_{eng}"] = stats["sharpe"]
        row[f"max_dd_pct_{eng}"] = stats["max_dd_pct"]
    return row


def _equity_stats(v: pd.Series) -> dict | None:
    """compute standard financial stats from an equity curve."""
    if len(v) < 2:
        return None
    total_ret = (v.iloc[-1] / v.iloc[0] - 1) * 100
    n_years = (v.index[-1] - v.index[0]).days / 365.25
    base = 1 + total_ret / 100
    cagr = (max(base, 0) ** (1 / max(n_years, 0.01)) - 1) * 100 if base > 0 else -100.0
    rets = v.pct_change().dropna()
    ann_vol = float(rets.std() * math.sqrt(252) * 100)
    daily_rf = 0.04 / 252
    excess = rets - daily_rf
    vol = float(rets.std())
    sharpe = float(excess.mean() / vol * math.sqrt(252)) if vol > 1e-10 else 0.0
    dd = ((v - v.cummax()) / v.cummax()).min() * 100
    return {
        "total_return_pct": total_ret,
        "cagr_pct": cagr,
        "ann_vol_pct": ann_vol,
        "sharpe": sharpe,
        "max_dd_pct": dd,
    }


def _es_row(bid: str, df: pd.DataFrame, engines: list[str], metrics: list[str], bucket_cccs: pd.DataFrame | None = None) -> dict:
    """compute ES, IUI, CSI for one benchmark; merge bucket-level CCC."""
    row: dict = {"benchmark_id": bid}
    for m in metrics:
        vals, matched_engines = _gather_metric(df, bid, m, engines)
        if len(vals) < 2:
            continue
        arr = np.array(vals)
        k = len(arr)
        mean_val = float(np.mean(arr))
        std_pop = float(np.std(arr, ddof=0))
        std_samp = float(np.std(arr, ddof=1))

        row[f"es_{m}"] = round(abs(std_pop / mean_val * 100) if abs(mean_val) > 1e-10 else 0.0, 4)
        row[f"es_range_{m}"] = round(float(arr.max() - arr.min()), 6)

        if k >= 3:
            t_crit = float(sp_stats.t.ppf(0.975, df=k - 1))
            row[f"iui_lower_{m}"] = round(mean_val - t_crit * std_samp, 6)
            row[f"iui_upper_{m}"] = round(mean_val + t_crit * std_samp, 6)
            row[f"iui_width_{m}"] = round(2 * t_crit * std_samp, 6)

        if m == "sharpe":
            row["csi_sharpe"] = int(arr.min() < 0 < arr.max())
            row["csi_sharpe_frac"] = round(float((arr < 0).sum()) / k, 4)

        for eng, val in zip(matched_engines, vals):
            row[f"{m}_{eng}"] = val

        if bucket_cccs is not None and bid in bucket_cccs.index:
            for col in (f"ccc_min_{m}", f"ccc_mean_{m}"):
                if col in bucket_cccs.columns:
                    row[col] = bucket_cccs.loc[bid, col]
    return row


def _add_daf(conc: pd.DataFrame, metrics: list[str], baseline: str = "01-equal-weight") -> None:
    """compute divergence amplification factor relative to baseline benchmark (in-place).

    only computed for metrics where the baseline ES range exceeds MIN_DAF_DENOM
    to avoid noise-amplified ratios (e.g. daf_ann_vol = 472x from a 0.003 pp denominator).
    """
    MIN_DAF_DENOM = 0.05
    if baseline not in conc.index:
        return
    for m in metrics:
        col = f"es_range_{m}"
        if col not in conc.columns:
            continue
        base_val = conc.loc[baseline, col]
        daf_col = f"daf_{m}"
        if abs(base_val) >= MIN_DAF_DENOM:
            conc[daf_col] = (conc[col] / base_val).round(4)
        else:
            conc[daf_col] = np.nan


def _gather_metric(df: pd.DataFrame, bid: str, metric: str, engines: list[str]):
    """collect values for one metric across available engines."""
    vals, eng_names = [], []
    for e in engines:
        col = f"{metric}_{e}"
        if col in df.columns and pd.notna(df.loc[bid, col]):
            vals.append(df.loc[bid, col])
            eng_names.append(e)
    return vals, eng_names


def _lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """compute lin's concordance correlation coefficient between two arrays."""
    if len(x) < 3 or len(y) < 3:
        return np.nan
    sx, sy = float(np.std(x, ddof=0)), float(np.std(y, ddof=0))
    if sx < 1e-12 or sy < 1e-12:
        return np.nan
    rho = float(np.corrcoef(x, y)[0, 1])
    mx, my = float(np.mean(x)), float(np.mean(y))
    return (2 * rho * sx * sy) / (sx**2 + sy**2 + (mx - my) ** 2)


def _compute_bucket_cccs(metrics: list[str]) -> pd.DataFrame:
    """per-benchmark CCC across 30 buckets for each engine pair and metric."""
    bucket_stats = collect_bucket_engine_stats()
    if bucket_stats.empty:
        return pd.DataFrame()
    rows = []
    for bid, bgroup in bucket_stats.groupby("benchmark_id"):
        row: dict = {"benchmark_id": bid}
        for metric in metrics:
            mdata = bgroup[bgroup["metric"] == metric]
            wide = mdata.pivot_table(
                index="bucket_id",
                columns="engine",
                values="value",
            )
            engines = sorted(wide.columns)
            cccs = []
            for ea, eb in itertools.combinations(engines, 2):
                pair = wide[[ea, eb]].dropna()
                ccc = _lin_ccc(pair[ea].values, pair[eb].values)
                if not np.isnan(ccc):
                    cccs.append(ccc)
            if cccs:
                row[f"ccc_min_{metric}"] = round(min(cccs), 4)
                row[f"ccc_mean_{metric}"] = round(float(np.mean(cccs)), 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("benchmark_id") if rows else pd.DataFrame()


# -- per-bucket engine stats for distribution plots ----------------------------


def collect_bucket_engine_stats(
    results_root: Path = RESULTS_ROOT,
    engines: tuple[str, ...] = ACTIVE_ENGINES,
) -> pd.DataFrame:
    """per-bucket, per-engine financial metrics in long form."""
    records = discover_benchmarks(results_root)
    rows: list[dict] = []
    for rec in records:
        bid = rec.get("benchmark_id")
        if bid is None:
            continue
        bdir = results_root / bid / "buckets"
        if bdir.exists():
            _scan_engine_buckets(rows, bid, bdir, engines)
    return pd.DataFrame(rows)


def _scan_engine_buckets(rows: list, bid: str, bdir: Path, engines: tuple) -> None:
    """iterate bucket directories under one benchmark."""
    for bp in sorted(bdir.iterdir()):
        _read_bucket_engines(rows, bid, bp, engines)


def _read_bucket_engines(rows: list, bid: str, bp: Path, engines: tuple) -> None:
    """compute per-engine stats from one bucket's equity.csv."""
    eqp = bp / "equity.csv"
    if not eqp.exists():
        return
    edf = pd.read_csv(eqp, index_col=0, parse_dates=True)
    for eng in engines:
        if eng not in edf.columns:
            continue
        stats = _equity_stats(edf[eng].dropna())
        if stats is not None:
            _append_engine_rows(rows, bid, bp.name, eng, stats)


def _append_engine_rows(
    rows: list,
    bid: str,
    bucket_id: str,
    eng: str,
    stats: dict,
) -> None:
    """append one row per metric from computed engine stats."""
    for m, v in stats.items():
        rows.append({"benchmark_id": bid, "bucket_id": bucket_id, "engine": eng, "metric": m, "value": v})
