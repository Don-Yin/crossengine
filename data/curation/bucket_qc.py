"""bucket-level quality control for 30 non-overlapping 6-stock buckets."""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from qc import NATURE_RC, SECTOR_COLORS, load_universe, sector_order

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
QC_DIR = DATA_DIR / "qc"
BUCKETS_PATH = DATA_DIR / "buckets.json"
CLOSE_PATH = DATA_DIR / "close.parquet"

TRADING_DAYS_PER_YEAR = 252
N_BUCKETS = 30
STOCKS_PER_BUCKET = 6


def load_buckets() -> list[dict]:
    """load and validate the structural integrity of buckets.json."""
    with open(BUCKETS_PATH) as f:
        buckets = json.load(f)
    required_keys = {"bucket_id", "tickers", "sectors"}
    for i, b in enumerate(buckets):
        missing = required_keys - set(b.keys())
        if missing:
            raise ValueError(f"bucket {i} missing keys: {missing}")
    return buckets


def verify_non_overlap(buckets: list[dict], all_tickers: list[str]) -> dict:
    """verify each stock appears exactly once across all buckets."""
    appearances: Counter = Counter()
    for b in buckets:
        appearances.update(b["tickers"])
    unique = len(appearances)
    overlap_pairs = sum(1 for a, b in combinations(
        [frozenset(b["tickers"]) for b in buckets], 2) if a & b)
    return {
        "unique_stocks": unique, "expected": len(all_tickers),
        "all_appear_once": all(appearances[t] == 1 for t in all_tickers),
        "overlap_pairs": overlap_pairs,
    }


def compute_sector_chi_square(buckets: list[dict], sectors: list[str]) -> dict:
    """chi-square goodness-of-fit for sector uniformity across bucket slots."""
    from scipy.stats import chi2 as chi2_dist
    sector_counts = Counter()
    for b in buckets:
        sector_counts.update(b["sectors"])
    total, n_s = sum(sector_counts.values()), len(sectors)
    expected = total / n_s
    chi2 = sum((sector_counts.get(s, 0) - expected) ** 2 / expected for s in sectors)
    p_value = 1 - chi2_dist.cdf(chi2, n_s - 1)
    return {"chi2_statistic": round(chi2, 4), "p_value": round(p_value, 4),
            "sector_counts": {s: sector_counts.get(s, 0) for s in sectors},
            "expected_per_sector": round(expected, 2)}


def compute_sector_entropy(buckets: list[dict], sectors: list[str]) -> dict:
    """Shannon entropy of sector distribution across bucket slots."""
    sector_counts = Counter()
    for b in buckets:
        sector_counts.update(b["sectors"])
    total = sum(sector_counts.values())
    probs = [sector_counts.get(s, 0) / total for s in sectors]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_ent = np.log2(len(sectors))
    return {"entropy": round(float(entropy), 4), "max_entropy": round(float(max_ent), 4),
            "normalized_entropy": round(float(entropy / max_ent), 4)}


def compute_bucket_characteristics(buckets: list[dict], close: pd.DataFrame) -> list[dict]:
    """compute per-bucket financial characteristics for diversity validation."""
    rets = close.pct_change().iloc[1:]
    return [_single_bucket_chars(b, rets, close) for b in buckets]


def _single_bucket_chars(bucket: dict, rets: pd.DataFrame, close: pd.DataFrame) -> dict:
    """compute characteristics for one bucket."""
    tickers = [t for t in bucket["tickers"] if t in rets.columns]
    sub = rets[tickers]
    vol = float(sub.std().mean() * np.sqrt(TRADING_DAYS_PER_YEAR))
    corr_mat = sub.corr()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape, dtype=bool), k=1)).stack()
    mean_corr = float(upper.mean()) if len(upper) > 0 else 0.0
    total_ret = float(((close[tickers].iloc[-1] / close[tickers].iloc[0]) - 1).mean() * 100)
    return {
        "bucket_id": bucket["bucket_id"], "tickers": tickers,
        "n_sectors": len(set(bucket["sectors"])),
        "mean_ann_vol": round(vol, 4), "mean_pairwise_corr": round(mean_corr, 4),
        "mean_total_return_pct": round(total_ret, 2),
    }


def _update_cooc_for_bucket(cooc: np.ndarray, bucket_sectors: list[str], idx: dict[str, int]) -> None:
    """update co-occurrence matrix for one bucket's sector pairs."""
    for s1, s2 in combinations(bucket_sectors, 2):
        i, j = idx[s1], idx[s2]
        cooc[i, j] += 1
        cooc[j, i] += 1


def _compute_sector_cooccurrence(buckets: list[dict], sectors: list[str]) -> np.ndarray:
    """count how many buckets each pair of sectors co-appears in."""
    idx = {s: i for i, s in enumerate(sectors)}
    cooc = np.zeros((len(sectors), len(sectors)), dtype=int)
    for b in buckets:
        _update_cooc_for_bucket(cooc, b["sectors"], idx)
    return cooc


def plot_sector_balance(chi2_result: dict, sectors: list[str], path: Path) -> None:
    """compact deviation dot plot with per-bucket sector composition strip."""
    import seaborn as sns

    counts = chi2_result["sector_counts"]
    expected = chi2_result["expected_per_sector"]
    labels = [s.replace("_", " ") for s in sectors]
    deviations = [counts.get(s, 0) - expected for s in sectors]
    colors = [SECTOR_COLORS.get(s, "#666") for s in sectors]

    with plt.rc_context(NATURE_RC):
        fig, (ax_dev, ax_strip) = plt.subplots(1, 2, figsize=(7.2, 2.8),
            gridspec_kw={"width_ratios": [1, 2.5]}, layout="constrained")

        y = np.arange(len(sectors))
        ax_dev.barh(y, deviations, color=colors, height=0.6, edgecolor="white", linewidth=0.3)
        ax_dev.axvline(0, color="#999999", linewidth=0.5)
        ax_dev.set_yticks(y)
        ax_dev.set_yticklabels(labels, fontsize=5.5)
        ax_dev.set_xlabel("deviation from expected", fontsize=6)
        ax_dev.set_title(f"sector balance (chi2={chi2_result['chi2_statistic']:.2f}, p={chi2_result['p_value']:.3f})",
                         fontsize=7)
        ax_dev.invert_yaxis()

        buckets = load_buckets()
        rows = []
        for b in buckets:
            bid = int(b["bucket_id"].replace("bucket-", ""))
            for sec in b["sectors"]:
                rows.append({"bucket": bid, "sector": sec.replace("_", " ")})
        df = pd.DataFrame(rows)
        palette = {s.replace("_", " "): SECTOR_COLORS[s] for s in sectors}
        sns.stripplot(data=df, x="bucket", y="sector", hue="sector", palette=palette,
                      order=labels, jitter=False, size=3, ax=ax_strip, legend=False)
        ax_strip.set_xlabel("bucket id", fontsize=6)
        ax_strip.set_ylabel("")
        ax_strip.set_title(f"sector composition per bucket ({N_BUCKETS} buckets x {STOCKS_PER_BUCKET} stocks)", fontsize=7)
        ax_strip.tick_params(axis="y", labelsize=5.5)
        ax_strip.tick_params(axis="x", labelsize=4.5)
        ax_strip.invert_yaxis()

        fig.savefig(path)
        plt.close(fig)
    print(f"  saved {path.name}")


def plot_diversity(chars: list[dict], path: Path) -> None:
    """scatter of per-bucket volatility vs correlation to verify balance."""
    vols = [c["mean_ann_vol"] for c in chars]
    corrs = [c["mean_pairwise_corr"] for c in chars]
    labels = [c["bucket_id"].replace("bucket-", "") for c in chars]
    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
        ax.scatter(vols, corrs, s=20, color="#0072B2", edgecolors="white", linewidth=0.3, zorder=5)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (vols[i], corrs[i]), fontsize=4.5, xytext=(2, 2), textcoords="offset points", color="#555555")
        ax.set_xlabel("mean annualized volatility")
        ax.set_ylabel("mean pairwise correlation")
        ax.set_title("bucket characteristic diversity (rerandomized)")
        ax.text(0.98, 0.98, f"vol: {min(vols):.3f} - {max(vols):.3f}\ncorr: {min(corrs):.3f} - {max(corrs):.3f}",
                transform=ax.transAxes, fontsize=5, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="#cccccc", linewidth=0.3))
        fig.savefig(path)
        plt.close(fig)
    print(f"  saved {path.name}")


def plot_sector_cooccurrence(cooc: np.ndarray, sectors: list[str], path: Path) -> None:
    """heatmap of sector pair co-occurrence within buckets."""
    labels = [s.replace("_", " ") for s in sectors]
    heatmap_rc = {**NATURE_RC, "axes.spines.top": True, "axes.spines.right": True}
    n = len(sectors)
    mask = np.eye(n, dtype=bool)
    display = cooc.astype(float)
    display[mask] = np.nan
    with plt.rc_context(heatmap_rc):
        fig, ax = plt.subplots(figsize=(4.5, 4.0), layout="constrained")
        cmap = plt.cm.Blues.copy()
        cmap.set_bad("#f0f0f0")
        ax.imshow(display, cmap=cmap, aspect="equal", interpolation="nearest")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)
        _annotate_cooccurrence(ax, cooc, n)
        off_diag = cooc[~mask]
        ax.set_title(f"sector co-occurrence within buckets (mean = {off_diag.mean():.1f})")
        fig.savefig(path)
        plt.close(fig)
    print(f"  saved {path.name}")


def _annotate_cooccurrence(ax, cooc: np.ndarray, n: int) -> None:
    """add text annotations to the co-occurrence heatmap cells."""
    for i, j in product(range(n), range(n)):
        if i == j:
            continue
        ax.text(j, i, str(cooc[i, j]), ha="center", va="center",
                fontsize=4.5, color="black" if cooc[i, j] < 8 else "white")


def main() -> None:
    """run all bucket-level quality checks."""
    print("loading data ...")
    universe = load_universe()
    buckets = load_buckets()
    assets = universe["assets"]
    all_tickers = sorted(assets.keys())
    sectors = sector_order(universe)
    print(f"  {len(buckets)} buckets, {len(all_tickers)} tickers, {len(sectors)} sectors\n")
    QC_DIR.mkdir(parents=True, exist_ok=True)

    print("computing diagnostics ...")
    overlap = verify_non_overlap(buckets, all_tickers)
    print(f"  non-overlap: unique={overlap['unique_stocks']}, all_once={overlap['all_appear_once']}, overlap_pairs={overlap['overlap_pairs']}")

    chi2_result = compute_sector_chi_square(buckets, sectors)
    print(f"  sector chi2: {chi2_result['chi2_statistic']:.2f}, p={chi2_result['p_value']:.3f}")

    entropy = compute_sector_entropy(buckets, sectors)
    print(f"  sector entropy: {entropy['entropy']:.4f} / {entropy['max_entropy']:.4f} (normalized = {entropy['normalized_entropy']:.4f})")

    cooc = _compute_sector_cooccurrence(buckets, sectors)

    chars = []
    if CLOSE_PATH.exists():
        close = pd.read_parquet(CLOSE_PATH).loc["2020-01-01":"2025-01-01"]
        chars = compute_bucket_characteristics(buckets, close)
        vols = [c["mean_ann_vol"] for c in chars]
        corrs = [c["mean_pairwise_corr"] for c in chars]
        print(f"  vol range:  {min(vols):.4f} - {max(vols):.4f} (var = {np.var(vols):.6f})")
        print(f"  corr range: {min(corrs):.4f} - {max(corrs):.4f} (var = {np.var(corrs):.6f})")
    else:
        print("  [skipping bucket characteristics -- close.parquet not found]")

    stats = {"n_buckets": len(buckets), "n_tickers": len(all_tickers), "n_sectors": len(sectors),
             "non_overlap": overlap, "sector_chi_square": chi2_result, "sector_entropy": entropy}
    if chars:
        stats["bucket_characteristics"] = chars
    (QC_DIR / "bucket-stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\n  saved bucket-stats.json")

    print("\ngenerating plots ...")
    plot_sector_balance(chi2_result, sectors, QC_DIR / "bucket-sector-balance.png")
    plot_sector_cooccurrence(cooc, sectors, QC_DIR / "bucket-sector-cooccurrence.png")
    if chars:
        plot_diversity(chars, QC_DIR / "bucket-diversity.png")
    print("\ndone.")


if __name__ == "__main__":
    main()
