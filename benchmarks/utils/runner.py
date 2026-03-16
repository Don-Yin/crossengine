"""multi-bucket benchmark runner -- orchestrates stratified bucket runs."""

from __future__ import annotations

import importlib.util
import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from utils._runner_agg import agg_divergence, collect_numeric, mean_dict, stats_dict
from utils.data import DATA_DIR, RESULTS_ROOT, load_close, load_close_full, load_spx

__all__ = ["run_multi_bucket", "load_buckets"]

# suppress loky semaphore-leak warnings from multiprocessing resource_tracker
# (benign: loky creates semaphores on import but doesn't unregister them;
# Python 3.12+ resource_tracker reports them as leaked at shutdown)
_rt_filter = "ignore::UserWarning:multiprocessing.resource_tracker"
_pw = os.environ.get("PYTHONWARNINGS", "")
if _rt_filter not in _pw:
    os.environ["PYTHONWARNINGS"] = f"{_pw},{_rt_filter}" if _pw else _rt_filter

BUCKETS_PATH = DATA_DIR / "buckets.json"
_worker_state: dict = {}


def load_buckets() -> list[dict]:
    """load bucket definitions from data/buckets.json."""
    return json.loads(BUCKETS_PATH.read_text())


def _spawn_init(benchmarks_dir: str, close_bytes: bytes, spx_bytes: bytes,
                buckets_json: str, n_assets: int) -> None:
    """called once per spawned worker -- loads shared data into process memory."""
    import pickle
    if benchmarks_dir not in sys.path:
        sys.path.insert(0, benchmarks_dir)
    os.environ["_BACKTEST_WORKER"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    logging.disable(logging.INFO)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    _worker_state["close_all"] = pickle.loads(close_bytes)
    _worker_state["spx"] = pickle.loads(spx_bytes)
    _worker_state["buckets"] = json.loads(buckets_json)
    _worker_state["n_assets"] = n_assets


def _spawn_run_bucket(args: tuple) -> str:
    """run one bucket in a spawned worker process."""
    script_path, bucket_idx, bucket_dir_str = args
    s = _worker_state
    bucket = s["buckets"][bucket_idx]
    tickers = bucket["tickers"][:s["n_assets"]]
    close = s["close_all"][tickers].dropna()
    spec = importlib.util.spec_from_file_location("_bm", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run_single(close, s["spx"], Path(bucket_dir_str))
    return bucket["bucket_id"]


class BucketRunner:
    """orchestrates running a benchmark across all stratified buckets."""

    def __init__(
        self,
        run_single: Callable[[pd.DataFrame, pd.Series, Path], None],
        results_dir: Path,
        n_assets: int = 6,
        full_history: bool = False,
        parallel_buckets: bool = True,
    ) -> None:
        self.run_single = run_single
        self.results_dir = results_dir
        self.n_assets = n_assets
        self.parallel_buckets = parallel_buckets
        self.buckets = load_buckets()
        self.close_all = load_close_full() if full_history else load_close()
        self.spx = load_spx()
        self.buckets_dir = results_dir / "buckets"

    def run(self) -> None:
        """execute all bucket runs then aggregate."""
        self.buckets_dir.mkdir(parents=True, exist_ok=True)
        if self.parallel_buckets:
            self._run_parallel()
        else:
            self._run_sequential()
        self._aggregate()

    def _run_sequential(self) -> None:
        """run buckets one by one (fallback)."""
        total = len(self.buckets)
        for i, bucket in enumerate(self.buckets, 1):
            tickers = bucket["tickers"][:self.n_assets]
            close = self.close_all[tickers].dropna()
            self.run_single(close, self.spx, self.buckets_dir / bucket["bucket_id"])
            logger.info("[%d/%d] %s", i, total, bucket["bucket_id"])

    def _run_parallel(self) -> None:
        """dispatch buckets to a spawn-based process pool (no fork, macOS safe)."""
        import pickle
        script_path = str(Path(self.run_single.__code__.co_filename))
        benchmarks_dir = str(Path(script_path).resolve().parent.parent)
        init_args = (
            benchmarks_dir,
            pickle.dumps(self.close_all),
            pickle.dumps(self.spx),
            json.dumps(self.buckets),
            self.n_assets,
        )
        tasks = [
            (script_path, i, str(self.buckets_dir / self.buckets[i]["bucket_id"]))
            for i in range(len(self.buckets))
        ]
        total = len(self.buckets)
        n_workers = min(os.cpu_count() or 4, total)
        logger.info("dispatching %d buckets across %d workers", total, n_workers)
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_spawn_init, initargs=init_args) as pool:
            for done, bid in enumerate(pool.imap_unordered(_spawn_run_bucket, tasks), 1):
                logger.info("[%d/%d] %s", done, total, bid)

    def _aggregate(self) -> None:
        """build aggregate metrics, equity, and plots from per-bucket results."""
        per_bucket = self._load_bucket_metrics()
        if not per_bucket:
            logger.warning("no bucket metrics found -- skipping aggregation")
            return
        engine_collected = collect_numeric(per_bucket, "engine_metrics")
        benchmark_id = str(self.results_dir.resolve().relative_to(RESULTS_ROOT.resolve()))
        payload = {
            "schema_version": 3, "benchmark_id": benchmark_id,
            "title": per_bucket[0].get("title", ""),
            "n_buckets": len(per_bucket),
            "engine_metrics": mean_dict(engine_collected),
            "benchmark_metrics": mean_dict(collect_numeric(per_bucket, "benchmark_metrics")),
            "asset_avg_metrics": mean_dict(collect_numeric(per_bucket, "asset_avg_metrics")),
            "divergence": agg_divergence(per_bucket),
            "population": stats_dict(engine_collected),
        }
        (self.results_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
        self._aggregate_equity()
        from utils._runner_plots import plot_distributions
        plot_distributions(self.results_dir / "plots", per_bucket)

    def _load_bucket_metrics(self) -> list[dict]:
        """read metrics.json from each bucket that exists."""
        return [
            json.loads(p.read_text())
            for b in self.buckets
            if (p := self.buckets_dir / b["bucket_id"] / "metrics.json").exists()
        ]

    def _aggregate_equity(self) -> None:
        """compute mean and std equity curves across buckets (inner join on dates)."""
        frames = self._load_equity_frames(100_000)
        if not frames:
            return
        combined = pd.concat(frames, axis=1, join="inner", keys=range(len(frames)))
        mean_eq = combined.T.groupby(level=1).mean().T
        std_eq = combined.T.groupby(level=1).std(ddof=1).T
        mean_eq.to_csv(self.results_dir / "equity.csv")
        std_eq.to_csv(self.results_dir / "equity_std.csv")

    def _load_equity_frames(self, initial: float) -> list[pd.DataFrame]:
        """load and rebase per-bucket equity csvs."""
        frames = []
        for bucket in self.buckets:
            epath = self.buckets_dir / bucket["bucket_id"] / "equity.csv"
            if not epath.exists():
                continue
            df = pd.read_csv(epath, index_col=0, parse_dates=True)
            frames.append(df / df.iloc[0] * initial)
        return frames


def run_multi_bucket(
    run_single: Callable[[pd.DataFrame, pd.Series, Path], None],
    results_dir: Path,
    n_assets: int = 6,
    full_history: bool = False,
    parallel_buckets: bool = True,
) -> None:
    """convenience wrapper -- create a BucketRunner and run it."""
    BucketRunner(run_single, results_dir, n_assets, full_history, parallel_buckets).run()
