"""run all (or selected) benchmarks sequentially.

usage:
    python benchmarks/run_all.py                    # run all
    python benchmarks/run_all.py 01 03 08           # run only matching prefixes
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
import traceback
from pathlib import Path

from utils.log import setup_logging

logger = logging.getLogger(__name__)

BENCHMARKS_DIR = Path(__file__).resolve().parent


def discover_benchmarks() -> list[tuple[str, Path]]:
    """find all benchmark run.py scripts, including 08-ml-signal sub-approaches."""
    entries: list[tuple[str, Path]] = []
    for d in sorted(BENCHMARKS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith(("_", ".")) or d.name in ("utils", "summary", "__pycache__"):
            continue
        run_py = d / "run.py"
        approaches = d / "approaches"
        if approaches.is_dir():
            for script in sorted(approaches.glob("*.py")):
                if script.name.startswith("_"):
                    continue
                entries.append((f"{d.name}/{script.stem}", script))
        elif run_py.exists():
            entries.append((d.name, run_py))
    return entries


def matches_filter(name: str, filters: list[str]) -> bool:
    """check if benchmark name matches any of the user-provided filter prefixes."""
    return any(name.startswith(f) or name.split("/")[0].startswith(f) for f in filters)


def run_one(name: str, script: Path) -> tuple[str, float, bool]:
    """import and run a single benchmark, return (name, elapsed_seconds, success)."""
    logger.info("=" * 60)
    logger.info("  %s", name)
    logger.info("=" * 60)
    t0 = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location(name, script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        elapsed = time.perf_counter() - t0
        return name, elapsed, True
    except Exception:
        elapsed = time.perf_counter() - t0
        traceback.print_exc()
        return name, elapsed, False


def run_summary() -> tuple[float, bool]:
    """run the cross-benchmark summary generator."""
    summary_script = BENCHMARKS_DIR / "summary" / "run.py"
    logger.info("=" * 60)
    logger.info("  summary")
    logger.info("=" * 60)
    t0 = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location("summary_run", summary_script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        elapsed = time.perf_counter() - t0
        return elapsed, True
    except Exception:
        elapsed = time.perf_counter() - t0
        traceback.print_exc()
        return elapsed, False


def main() -> None:
    """entry point: parse args, discover benchmarks, run sequentially."""
    parser = argparse.ArgumentParser(description="run all or selected benchmarks")
    parser.add_argument("benchmarks", nargs="*", help="prefix filters (e.g. 01 03 08)")
    args = parser.parse_args()

    all_benchmarks = discover_benchmarks()
    selected = [(n, p) for n, p in all_benchmarks if matches_filter(n, args.benchmarks)] if args.benchmarks else all_benchmarks

    if not selected:
        logger.error("no benchmarks matched filters: %s", args.benchmarks)
        logger.info("available: %s", [n for n, _ in all_benchmarks])
        sys.exit(1)

    logger.info("running %d benchmark(s):", len(selected))
    for name, _ in selected:
        logger.info("  %s", name)

    results: list[tuple[str, float, bool]] = []
    t_total = time.perf_counter()

    for name, script in selected:
        name, elapsed, ok = run_one(name, script)
        results.append((name, elapsed, ok))
        tag = "OK" if ok else "FAIL"
        logger.info("[%s] %s in %.0fs", name, tag, elapsed)

    summary_elapsed, summary_ok = run_summary()
    summary_tag = "OK" if summary_ok else "FAIL"
    logger.info("[summary] %s in %.0fs", summary_tag, summary_elapsed)

    wall = time.perf_counter() - t_total
    n_fail = sum(1 for _, _, ok in results if not ok) + (0 if summary_ok else 1)
    logger.info("=" * 60)
    logger.info("all done: %d benchmarks + summary in %.0fs (%d failed)", len(results), wall, n_fail)
    logger.info("=" * 60)
    for name, elapsed, ok in results:
        status = "OK" if ok else "FAIL"
        logger.info("  %s %7.0fs  %s", status, elapsed, name)
    logger.info("  %s %7.0fs  summary", summary_tag, summary_elapsed)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    setup_logging()
    main()
