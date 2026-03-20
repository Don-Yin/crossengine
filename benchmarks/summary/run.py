"""Generate all cross-benchmark summary outputs.

Usage:
    python benchmarks/summary/run.py

Discovers every benchmark that has a ``metrics.json`` in .results/,
collects them into a single table, and generates all summary figures.
New benchmarks and new figures are picked up automatically.
"""

from __future__ import annotations

from summary.collect import (
    collect_metrics, write_summary_csv, write_benchmark_relative_csv,
    write_asset_avg_csv, SUMMARY_DIR,
)
from summary.concordance import compute_engine_concordance
from summary.concordance_diagnostics import (
    compute_bucket_autocorrelation,
    compute_bucket_exchangeability,
    compute_floor_decomposition,
)
from summary import figures


def _save_intermediate_json(df, conc) -> None:
    """persist intermediate analysis data as json for reuse."""
    import json
    import numpy as np
    from summary.collect import TABLES_RAW_DIR
    ir_dir = TABLES_RAW_DIR / "implementation-risk"
    ir_dir.mkdir(parents=True, exist_ok=True)

    cols = ["num_trades", "total_commissions", "total_slippage",
            "div_bt_max_rel_pct", "div_vectorbt_max_rel_pct",
            "ann_volatility_pct", "max_drawdown_pct"]
    keep = [c for c in cols if c in df.columns]
    sub = df[keep].copy()
    sub["total_cost"] = sub.get("total_commissions", 0) + sub.get("total_slippage", 0)
    sub["per_trade_cost"] = sub["total_cost"] / sub["num_trades"].replace(0, np.nan)
    p1 = ir_dir / "complexity-data.json"
    sub.reset_index().to_json(p1, orient="records", indent=2, default_handler=str)
    print(f"  json -> {p1}")

    if not conc.empty:
        p2 = ir_dir / "concordance-full.json"
        conc.reset_index().to_json(p2, orient="records", indent=2, default_handler=str)
        print(f"  json -> {p2}")


def _save_icc_and_floor() -> None:
    """persist icc, exchangeability, and floor decomposition tables."""
    from summary.collect import TABLES_RAW_DIR
    ir_dir = TABLES_RAW_DIR / "implementation-risk"
    ir_dir.mkdir(parents=True, exist_ok=True)

    exc_detail, exc_summary, exc_extra = compute_bucket_exchangeability()
    if not exc_summary.empty:
        p = ir_dir / "exchangeability-summary.csv"
        exc_summary.to_csv(p, index=False)
        print(f"  table -> {p}")
        for _, row in exc_summary.iterrows():
            print(f"    cluster-bootstrap: rho = {row['observed_rho']:.4f} "
                  f"95% CI [{row['bootstrap_ci95_lower']:.4f}, {row['bootstrap_ci95_upper']:.4f}] "
                  f"-> {row['interpretation']}")

    icc_detail, cat_summary = compute_bucket_autocorrelation()
    if not icc_detail.empty:
        p = ir_dir / "icc-detail.csv"
        icc_detail.to_csv(p, index=False)
        print(f"  table -> {p}")
    if not cat_summary.empty:
        p = ir_dir / "icc-category.csv"
        cat_summary.to_csv(p, index=False)
        print(f"  table -> {p}")
        for _, row in cat_summary.iterrows():
            print(f"    {row['category']}: mean lag-1 autocorr = {row['mean_lag1_ac']:.4f} "
                  f"[{row['ci95_lower']:.4f}, {row['ci95_upper']:.4f}] "
                  f"(deprecated)")

    floor_df = compute_floor_decomposition()
    if not floor_df.empty:
        p = ir_dir / "floor-decomposition.csv"
        floor_df.to_csv(p, index=False)
        print(f"  table -> {p}")


def main():
    """generate all summary tables and figures."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    df = collect_metrics()
    if df.empty:
        print("no metrics.json files found -- run benchmarks first")
        return

    from summary.figures._common import register_n_buckets
    register_n_buckets(df)

    print(f"discovered {len(df)} benchmarks\n")
    print(df[["title"]].to_string())
    print()

    print("writing tables...\n")
    write_summary_csv()
    write_benchmark_relative_csv()
    write_asset_avg_csv()

    from summary.collect import TABLES_RAW_DIR
    conc = compute_engine_concordance()
    if not conc.empty:
        conc_path = TABLES_RAW_DIR / "engine-concordance.csv"
        conc.to_csv(conc_path)
        print(f"  table -> {conc_path}")

    print("\ngenerating figures...\n")
    figures.plot_divergence_landscape(df)
    figures.plot_divergence_vs_cost(df)
    figures.plot_divergence_timeseries(df)
    figures.plot_risk_return(df)
    figures.plot_performance_heatmap(df)
    figures.plot_divergence_correlation(df)
    figures.plot_equity_overlay(df)
    figures.plot_drawdown_comparison(df)
    figures.plot_engine_concordance(df, conc=conc)
    figures.plot_engine_agreement(df, conc=conc)

    print("\ngenerating implementation risk figures...\n")
    _save_intermediate_json(df, conc)
    figures.plot_metric_sensitivity(df, conc=conc)
    figures.plot_economic_significance(df)
    figures.plot_complexity_analysis(df)
    figures.plot_divergence_anatomy(df)
    figures.plot_mechanism_comparison(df)

    print("\ngenerating visual tables...\n")
    figures.plot_table_raw(df)
    figures.plot_controlled_distributions(df)
    figures.plot_table_costs(df)

    print("\ngenerating profiling figures...\n")
    from summary.profiling_collect import collect_profiling
    df_prof = collect_profiling()
    if df_prof.empty:
        print("  no profiling.json files found -- skipping")
    else:
        print(f"  collected {len(df_prof)} profiling records "
              f"({df_prof['engine'].nunique()} engines, "
              f"{df_prof['bucket_id'].nunique()} buckets)")
        figures.plot_profiling_distributions(df_prof)

    print("\ngenerating statistical robustness figures...\n")
    figures.plot_tost_equivalence()
    figures.plot_power_analysis()
    figures.plot_wilcoxon_robustness()
    figures.plot_permutation_test()
    figures.plot_qq_normality()

    print("\ncomputing icc and floor decomposition...\n")
    _save_icc_and_floor()

    print("\ndone.")


if __name__ == "__main__":
    main()
