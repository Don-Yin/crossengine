"""generate 30 non-overlapping sector-balanced 6-stock buckets via rerandomization.

uses Morgan & Rubin (2012) rerandomization: generate many candidate
random partitions, score each by balance criterion, keep the best.
each stock appears in exactly 1 bucket; each bucket has 6 distinct sectors.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

N_BUCKETS = 30
STOCKS_PER_BUCKET = 6
N_CANDIDATES = 20_000
TRADING_DAYS_PER_YEAR = 252


class PartitionGenerator:
    """generates random non-overlapping partitions of stocks into sector-balanced buckets."""

    def __init__(self, stocks_by_sector: dict[str, list[str]]):
        self.stocks_by_sector = stocks_by_sector
        self.sectors = list(stocks_by_sector.keys())

    def generate(self, rng: random.Random) -> list[list[str]] | None:
        """attempt one random non-overlapping partition into N_BUCKETS buckets of 6."""
        buckets: list[list[str]] = [[] for _ in range(N_BUCKETS)]
        bucket_sectors: list[set[str]] = [set() for _ in range(N_BUCKETS)]
        order = rng.sample(self.sectors, len(self.sectors))
        shuffled = {s: rng.sample(self.stocks_by_sector[s], len(self.stocks_by_sector[s])) for s in self.sectors}
        for sector in order:
            eligible = self._eligible_buckets(sector, bucket_sectors, buckets)
            if len(eligible) < len(shuffled[sector]):
                return None
            rng.shuffle(eligible)
            eligible.sort(key=lambda i: len(buckets[i]))
            self._assign_stocks(shuffled[sector], eligible, buckets, bucket_sectors, sector)
        if any(len(b) != STOCKS_PER_BUCKET for b in buckets):
            return None
        return buckets

    def _eligible_buckets(self, sector: str, bucket_sectors: list[set], buckets: list[list]) -> list[int]:
        """find bucket indices that can accept a stock from the given sector."""
        return [i for i in range(N_BUCKETS) if sector not in bucket_sectors[i] and len(buckets[i]) < STOCKS_PER_BUCKET]

    def _assign_stocks(
        self, stocks: list[str], eligible: list[int],
        buckets: list[list], bucket_sectors: list[set], sector: str,
    ) -> None:
        """assign each stock to the next eligible bucket."""
        for stock, idx in zip(stocks, eligible[:len(stocks)]):
            buckets[idx].append(stock)
            bucket_sectors[idx].add(sector)


class BalanceScorer:
    """scores partition balance via Mahalanobis distance with fixed reference covariance."""

    def __init__(self, close: pd.DataFrame):
        """precompute stock-level covariates and fixed reference covariance (Morgan & Rubin 2012)."""
        returns = close.pct_change().iloc[1:]
        tickers = list(close.columns)
        vol = (returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)).to_dict()
        corr = returns.corr()
        total_ret = ((close.iloc[-1] / close.iloc[0]) - 1).to_dict()
        self._stock_x = {}
        for t in tickers:
            avg_corr = float(corr[t].drop(t).mean())
            log_ret = float(np.log(1 + abs(total_ret[t])))
            self._stock_x[t] = np.array([vol[t], avg_corr, log_ret])
        stock_mat = np.array([self._stock_x[t] for t in tickers])
        cov_pop = np.cov(stock_mat.T, ddof=0)
        self._ref_cov_inv = np.linalg.inv(cov_pop / STOCKS_PER_BUCKET + 1e-10 * np.eye(3))

    def score(self, partition: list[list[str]]) -> float:
        """mahalanobis balance: lower = more balanced; uses fixed reference covariance."""
        z = np.array([np.mean([self._stock_x[t] for t in b], axis=0) for b in partition])
        z_c = z - z.mean(axis=0)
        return float(np.sum((z_c @ self._ref_cov_inv) * z_c))


class Rerandomizer:
    """find the best partition via rerandomization (Morgan & Rubin 2012)."""

    def __init__(self, generator: PartitionGenerator, scorer: BalanceScorer):
        self.generator = generator
        self.scorer = scorer

    def run(self, n_candidates: int = N_CANDIDATES, base_seed: int = 42) -> tuple[list[list[str]], float, int, int]:
        """return (best_partition, best_score, best_seed, n_valid)."""
        best_partition, best_score, best_seed = None, float("inf"), base_seed
        n_valid = 0
        for i in range(n_candidates):
            seed = base_seed + i
            partition = self.generator.generate(random.Random(seed))
            if partition is None:
                continue
            n_valid += 1
            score = self.scorer.score(partition)
            if score < best_score:
                best_partition, best_score, best_seed = partition, score, seed
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{n_candidates} evaluated, best score = {best_score:.6f}")
        if best_partition is None:
            raise RuntimeError("no valid partition found in any candidate")
        return best_partition, best_score, best_seed, n_valid


def _load_universe(path: Path) -> dict:
    """load universe.json and return the full dict."""
    return json.loads(path.read_text())


def _group_by_sector(assets: dict) -> dict[str, list[str]]:
    """group tickers by GICS sector."""
    sectors: dict[str, list[str]] = {}
    for ticker, info in assets.items():
        sectors.setdefault(info["sector"], []).append(ticker)
    return sectors


def _build_sector_map(assets: dict) -> dict[str, str]:
    """build {ticker: sector} lookup from assets dict."""
    return {t: info["sector"] for t, info in assets.items()}


def _count_pairwise_overlaps(bucket_sets: list[frozenset]) -> int:
    """count the number of bucket pairs that share at least one stock."""
    return sum(1 for a, b in combinations(bucket_sets, 2) if a & b)


def _check_bucket_sectors(bucket: dict, ticker_to_sector: dict[str, str]) -> None:
    """validate that a single bucket has 6 distinct sectors."""
    secs = {ticker_to_sector[t] for t in bucket["tickers"]}
    if len(secs) != STOCKS_PER_BUCKET:
        raise ValueError(f'{bucket["bucket_id"]} has {len(secs)} sectors (need {STOCKS_PER_BUCKET})')


def format_buckets(partition: list[list[str]], ticker_to_sector: dict[str, str]) -> list[dict]:
    """convert partition into the output schema for buckets.json."""
    out = []
    for i, tickers in enumerate(partition, 1):
        secs = [ticker_to_sector[t] for t in tickers]
        out.append({
            "bucket_id": f"bucket-{i:02d}", "tickers": tickers, "sectors": secs,
            "description": f"non-overlapping sector-balanced bucket ({len(set(secs))} sectors)",
        })
    return out


def validate_and_report(buckets: list[dict], all_tickers: list[str], ticker_to_sector: dict[str, str]) -> None:
    """validate non-overlapping constraints and print summary."""
    if len(buckets) != N_BUCKETS:
        raise ValueError(f"expected {N_BUCKETS} buckets, got {len(buckets)}")
    appearances: Counter = Counter()
    for b in buckets:
        appearances.update(b["tickers"])
    if len(appearances) != len(all_tickers):
        raise ValueError(f"expected {len(all_tickers)} unique stocks, got {len(appearances)}")
    for t in all_tickers:
        if appearances[t] != 1:
            raise ValueError(f"{t} appears {appearances[t]} times (must be exactly 1)")
    for b in buckets:
        _check_bucket_sectors(b, ticker_to_sector)
    overlap_count = _count_pairwise_overlaps([frozenset(b["tickers"]) for b in buckets])
    if overlap_count:
        raise ValueError(f"{overlap_count} bucket pairs share stocks")
    sector_counts: Counter = Counter()
    for b in buckets:
        sector_counts.update(b["sectors"])
    print(f"buckets: {len(buckets)}, unique stocks: {len(appearances)}, each appears 1x")
    print(f"all buckets have {STOCKS_PER_BUCKET} distinct sectors, zero pairwise overlap")
    for sec in sorted(sector_counts):
        print(f"  {sec}: {sector_counts[sec]} bucket-appearances")
    print("all constraints satisfied.")


def main() -> None:
    """run rerandomization and generate buckets.json."""
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent
    close_path = data_dir / "close.parquet"
    output_path = data_dir / "buckets.json"
    qc_path = data_dir / "qc" / "stratification-stats.json"

    universe = _load_universe(data_dir / "universe.json")
    assets = universe["assets"]
    stocks_by_sector = _group_by_sector(assets)
    ticker_to_sector = _build_sector_map(assets)
    all_tickers = list(assets.keys())

    print(f"universe: {len(all_tickers)} stocks, {len(stocks_by_sector)} sectors")
    for sec, tickers in stocks_by_sector.items():
        print(f"  {sec}: {len(tickers)}")

    close = pd.read_parquet(close_path)
    close = close.loc["2020-01-01":"2025-01-01", all_tickers]
    print(f"\nclose data: {close.shape[0]} days x {close.shape[1]} stocks")
    print(f"\nrunning rerandomization with {N_CANDIDATES} candidates ...")

    generator = PartitionGenerator(stocks_by_sector)
    scorer = BalanceScorer(close)
    best_partition, best_score, best_seed, n_valid = Rerandomizer(generator, scorer).run()
    print(f"\nbest balance score: {best_score:.6f} (seed {best_seed})")

    buckets = format_buckets(best_partition, ticker_to_sector)
    validate_and_report(buckets, all_tickers, ticker_to_sector)
    output_path.write_text(json.dumps(buckets, indent=2) + "\n")
    print(f"\nwritten to {output_path}")

    qc_stats = {
        "n_candidates": N_CANDIDATES, "n_valid": n_valid,
        "acceptance_rate": round(n_valid / N_CANDIDATES, 4),
        "best_score": round(best_score, 6), "best_seed": best_seed,
        "n_buckets": N_BUCKETS, "stocks_per_bucket": STOCKS_PER_BUCKET,
        "covariates_used": ["mean_ann_vol", "mean_pairwise_corr", "log_total_return"],
        "balance_method": "mahalanobis_rerandomization",
        "reference": "Morgan & Rubin (2012). rerandomization to improve covariate balance."
        " Annals of Statistics 40(2), 1263-1282.",
    }
    qc_path.write_text(json.dumps(qc_stats, indent=2) + "\n")
    print(f"qc stats written to {qc_path}")
    print("run `python data/curation/bucket_qc.py` for bucket-level diagnostics.")


if __name__ == "__main__":
    main()
