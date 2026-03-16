"""WeightSchedule type, engine adapters, and unified benchmark runner.

A WeightSchedule is the universal interchange format between strategies
and engine adapters:  {date: {asset: target_weight}}.

individual engine adapters live in utils/wrappers/. this module re-exports
them and provides the unified run_benchmark() orchestrator.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from utils.types import WeightSchedule
from utils.wrappers import (
    run_backtrader_engine,
    run_bt_engine,
    run_cvxportfolio_engine,
    run_nautilus_engine,
    run_ours,
    run_vbt_engine,
    run_zipline_engine,
)

logger = logging.getLogger(__name__)

__all__ = [
    "WeightSchedule",
    "run_ours",
    "run_bt_engine",
    "run_vbt_engine",
    "run_backtrader_engine",
    "run_cvxportfolio_engine",
    "run_zipline_engine",
    "run_nautilus_engine",
    "run_bt",
    "run_vectorbt",
    "run_benchmark",
]

_pool: ProcessPoolExecutor | None = None


def _worker_init():
    """suppress progress-bar noise in engine worker processes."""
    os.environ["TQDM_DISABLE"] = "1"
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)


def _get_pool(max_workers: int = 5) -> ProcessPoolExecutor:
    """lazily create a shared process pool; reused across all buckets."""
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init)
        atexit.register(_pool.shutdown, wait=False)
        logger.info("process pool created (workers=%d)", max_workers)
    return _pool


def _timed_call(fn, kwargs: dict):
    """run fn(**kwargs) in a worker process; returns (result, wall_seconds)."""
    t0 = time.perf_counter()
    result = fn(**kwargs)
    return result, time.perf_counter() - t0


# ── old-style runners (used by benchmarks 01-04) ─────────────────────────


def run_bt(
    close: pd.DataFrame,
    strategy,
    *,
    initial_cash: float,
    commission_rate: float,
) -> pd.Series:
    """Run a pre-built bt.Strategy, return portfolio value series ($)."""
    import bt as _bt

    comm_fn = (lambda q, p: abs(q) * p * commission_rate) if commission_rate else (lambda q, p: 0)
    test = _bt.Backtest(
        strategy,
        close,
        initial_capital=initial_cash,
        commissions=comm_fn,
        integer_positions=False,
    )
    res = _bt.run(test)
    name = list(res.keys())[0]
    rebased = res.prices[name]
    return rebased / rebased.iloc[0] * initial_cash


def run_vectorbt(
    close: pd.DataFrame,
    size: pd.DataFrame,
    *,
    initial_cash: float,
    commission_rate: float,
) -> pd.Series:
    """Run vectorbt from a pre-built size DataFrame, return value series ($)."""
    import vectorbt as vbt

    pf = vbt.Portfolio.from_orders(
        close,
        size,
        size_type="targetpercent",
        fees=commission_rate,
        init_cash=initial_cash,
        freq="1D",
        group_by=True,
        cash_sharing=True,
        call_seq="auto",
    )
    return pf.value()


# ── unified benchmark runner ──────────────────────────────────────────────


def run_benchmark(
    ws: WeightSchedule,
    close: pd.DataFrame,
    *,
    results_dir: Path,
    title: str,
    initial_cash: float = 100_000,
    commission: float = 0.0015,
    slippage: float = 0.0003,
    note: str | None = None,
    spx: pd.Series | None = None,
    parallel: bool = True,
) -> None:
    """run all engines on one WeightSchedule, compare, write report."""
    from utils.comparison import write_comparison

    if not ws:
        warnings.warn(f"empty weight schedule for '{title}' -- all-cash portfolio")

    total_cost_rate = commission + slippage
    short_name = Path(results_dir).name
    tasks = {
        "ours": (run_ours, dict(close=close, ws=ws, initial_cash=initial_cash, commission=commission, slippage=slippage)),
        "bt": (run_bt_engine, dict(close=close, ws=ws, name=short_name, initial_cash=initial_cash, commission=total_cost_rate)),
        "vectorbt": (run_vbt_engine, dict(close=close, ws=ws, initial_cash=initial_cash, commission=total_cost_rate)),
        "backtrader": (run_backtrader_engine, dict(close=close, ws=ws, initial_cash=initial_cash, commission=total_cost_rate)),
        "cvxportfolio": (run_cvxportfolio_engine, dict(close=close, ws=ws, initial_cash=initial_cash, commission=total_cost_rate)),
    }

    in_worker = os.environ.get("_BACKTEST_WORKER") == "1"
    results, timings = _run_parallel(tasks) if (parallel and not in_worker) else _run_sequential(tasks)
    _save_timings(timings, results_dir)

    from utils.data import compute_asset_avg

    asset_avg = compute_asset_avg(close, initial_cash)

    write_comparison(
        results["ours"],
        {k: v for k, v in results.items() if k != "ours"},
        results_dir,
        title,
        note=note,
        spx=spx,
        asset_avg=asset_avg,
    )


def _run_parallel(tasks: dict) -> tuple[dict, dict]:
    """dispatch all engines to process pool, log progress as each completes."""
    pool = _get_pool()
    n = len(tasks)
    logger.info("dispatching %d engines in parallel", n)
    wall_t0 = time.perf_counter()

    futures = {pool.submit(_timed_call, fn, kwargs): name for name, (fn, kwargs) in tasks.items()}
    results, timings = {}, {}
    for i, future in enumerate(as_completed(futures), 1):
        name = futures[future]
        result, elapsed = future.result()
        results[name] = result
        timings[name] = {"wall_time_s": round(elapsed, 4)}
        logger.info("[%d/%d] %s finished (%.2fs)", i, n, name, elapsed)

    wall = time.perf_counter() - wall_t0
    logger.info("all %d engines done in %.2fs wall (%.1fx vs sequential)", n, wall, sum(t["wall_time_s"] for t in timings.values()) / max(wall, 0.01))
    return results, timings


def _run_sequential(tasks: dict) -> tuple[dict, dict]:
    """run engines one by one (fallback)."""
    results, timings = {}, {}
    n = len(tasks)
    for i, (name, (fn, kwargs)) in enumerate(tasks.items(), 1):
        t0 = time.perf_counter()
        results[name] = fn(**kwargs)
        elapsed = time.perf_counter() - t0
        timings[name] = {"wall_time_s": round(elapsed, 4)}
        logger.info("[%d/%d] %s finished (%.2fs)", i, n, name, elapsed)
    return results, timings


def _save_timings(timings: dict, results_dir: Path) -> None:
    """write profiling.json with wall times."""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "profiling.json").write_text(json.dumps(timings, indent=2))
