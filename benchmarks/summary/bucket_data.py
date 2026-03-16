"""collect per-bucket metrics for distribution visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from summary.collect import discover_benchmarks
from utils.data import RESULTS_ROOT

_SECTION_KEYS = {
    "engine": "engine_metrics",
    "spx": "benchmark_metrics",
    "asset_avg": "asset_avg_metrics",
}


class BucketCollector:
    """gathers per-bucket metrics from all benchmarks into long-form data."""

    def __init__(self, results_root: Path = RESULTS_ROOT):
        """initialize collector, discovering all benchmarks on construction."""
        self.results_root = results_root
        self._records = discover_benchmarks(results_root)

    def collect(self, sections: tuple[str, ...] = ("engine",)) -> pd.DataFrame:
        """return (benchmark_id, bucket_id, section, metric, value) long-form dataframe."""
        rows: list[dict] = []
        for rec in self._records:
            bid = rec.get("benchmark_id")
            if bid is None:
                continue
            bdir = self.results_root / bid / "buckets"
            if bdir.exists():
                self._scan_benchmark(rows, bid, bdir, sections)
        return pd.DataFrame(rows)

    def _scan_benchmark(self, rows: list, bid: str, bdir: Path, sections: tuple) -> None:
        """iterate buckets under one benchmark, delegating to _read_bucket."""
        for bp in sorted(bdir.iterdir()):
            self._read_bucket(rows, bid, bp, sections)

    def _read_bucket(self, rows: list, bid: str, bp: Path, sections: tuple) -> None:
        """extract requested sections from one bucket's metrics.json."""
        mp = bp / "metrics.json"
        if not mp.exists():
            return
        data = json.loads(mp.read_text())
        bucket_id = bp.name
        for section in sections:
            raw = self._resolve_section(data, section)
            prefix = f"{section}_" if section.startswith("div_") else ""
            _append_numeric(rows, bid, bucket_id, section, raw, prefix)

    @staticmethod
    def _resolve_section(data: dict, section: str) -> dict:
        """map a section name to the appropriate sub-dict in metrics.json."""
        if section.startswith("div_"):
            return data.get("divergence", {}).get(section[4:], {})
        return data.get(_SECTION_KEYS.get(section, section), {})


def _append_numeric(
    rows: list, bid: str, bucket_id: str, section: str,
    metrics: dict, prefix: str = "",
) -> None:
    """add one row per numeric metric value."""
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            rows.append({
                "benchmark_id": bid, "bucket_id": bucket_id,
                "section": section, "metric": f"{prefix}{k}", "value": v,
            })


def collect_bucket_metrics(
    results_root: Path = RESULTS_ROOT,
    sections: tuple[str, ...] = ("engine",),
) -> pd.DataFrame:
    """convenience wrapper around BucketCollector."""
    return BucketCollector(results_root).collect(sections)
