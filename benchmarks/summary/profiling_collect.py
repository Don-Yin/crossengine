"""collect profiling data from all benchmark bucket runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils.data import RESULTS_ROOT


def collect_profiling(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """walk all buckets, read profiling.json, return long-form DataFrame.

    columns: benchmark_id, bucket_id, engine, wall_time_s
    engine names are auto-discovered from json keys.
    """
    rows = []
    for prof_path in sorted(results_root.rglob("profiling.json")):
        parts = prof_path.relative_to(results_root).parts
        if "buckets" not in parts:
            continue
        bucket_idx = parts.index("buckets")
        benchmark_id = "/".join(parts[:bucket_idx])
        bucket_id = parts[bucket_idx + 1]

        data = json.loads(prof_path.read_text())
        for engine, stats in data.items():
            row = {
                "benchmark_id": benchmark_id,
                "bucket_id": bucket_id,
                "engine": engine,
            }
            row.update(stats)
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()
