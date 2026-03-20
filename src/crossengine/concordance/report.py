"""concordance report: holds multi-engine results and divergence metrics."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


class ConcordanceReport:
    """result of running a strategy through multiple engines."""

    def __init__(
        self,
        equity: dict[str, pd.Series],
        initial_cash: float,
    ) -> None:
        self._equity = equity
        self._initial_cash = initial_cash
        self._engines = sorted(equity.keys())
        self._pairs = list(combinations(self._engines, 2))
        self._divergence = self._compute_divergence()

    def _compute_divergence(self) -> dict[str, dict[str, float]]:
        """compute pairwise divergence metrics."""
        result = {}
        for a, b in self._pairs:
            sa, sb = self._equity[a], self._equity[b]
            common = sa.index.intersection(sb.index)
            va, vb = sa.loc[common].values, sb.loc[common].values
            midpoint = (va + vb) / 2.0
            abs_diff = np.abs(va - vb)
            rel_diff = np.where(midpoint > 0, abs_diff / midpoint * 100, 0.0)
            corr = np.corrcoef(va, vb)[0, 1] if len(va) > 1 else 1.0
            result[f"{a}_vs_{b}"] = {
                "max_abs_diff": float(np.max(abs_diff)),
                "max_rel_diff_pct": float(np.max(rel_diff)),
                "mean_rel_diff_pct": float(np.mean(rel_diff)),
                "correlation": float(corr),
            }
        return result

    @property
    def engines(self) -> list[str]:
        """list of engines that ran."""
        return list(self._engines)

    @property
    def equity(self) -> pd.DataFrame:
        """equity curves as a DataFrame (columns = engine names)."""
        return pd.DataFrame(self._equity)

    @property
    def divergence(self) -> dict[str, dict[str, float]]:
        """pairwise divergence metrics."""
        return dict(self._divergence)

    @property
    def max_divergence(self) -> float:
        """maximum relative divergence across all engine pairs (%)."""
        if not self._divergence:
            return 0.0
        return max(d["max_rel_diff_pct"] for d in self._divergence.values())

    @property
    def engine_sensitivity(self) -> float:
        """engine sensitivity (ES): coefficient of variation of final values across engines."""
        finals = [self._equity[e].iloc[-1] for e in self._engines]
        mean = np.mean(finals)
        if mean == 0:
            return 0.0
        return float(np.std(finals) / mean * 100)

    def summary(self) -> str:
        """human-readable concordance summary."""
        lines = [
            "engine concordance report",
            "=" * 50,
            f"engines: {', '.join(self._engines)} ({len(self._engines)} active)",
            "",
            "pairwise divergence (max relative %):",
        ]
        for pair, metrics in self._divergence.items():
            name = pair.replace("_vs_", " vs ")
            lines.append(f"  {name:<30} {metrics['max_rel_diff_pct']:>8.4f}%")
        lines.append("")
        lines.append(f"max divergence:      {self.max_divergence:.4f}%")
        lines.append(f"engine sensitivity:  {self.engine_sensitivity:.4f}%")
        lines.append("")
        lines.append("final portfolio values:")
        for e in self._engines:
            final = self._equity[e].iloc[-1]
            lines.append(f"  {e:<20} ${final:>12,.2f}")
        return "\n".join(lines)

    def to_json(self, path: str | Path) -> None:
        """write report to JSON file."""
        data = {
            "engines": self._engines,
            "initial_cash": self._initial_cash,
            "divergence": self._divergence,
            "max_divergence_pct": self.max_divergence,
            "engine_sensitivity_pct": self.engine_sensitivity,
            "final_values": {e: float(self._equity[e].iloc[-1]) for e in self._engines},
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    def plot(self):
        """plot equity curves for all engines."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")

        for e in self._engines:
            axes[0].plot(self._equity[e].index, self._equity[e].values, label=e, linewidth=1.2)
        axes[0].set_ylabel("portfolio value ($)")
        axes[0].set_title("engine concordance: equity curves")
        axes[0].legend()

        if len(self._engines) >= 2:
            ref = self._engines[0]
            ref_vals = self._equity[ref]
            for e in self._engines[1:]:
                common = ref_vals.index.intersection(self._equity[e].index)
                diff_pct = (self._equity[e].loc[common] - ref_vals.loc[common]) / ref_vals.loc[common] * 100
                axes[1].plot(common, diff_pct.values, label=f"{e} vs {ref}", linewidth=1)
            axes[1].axhline(0, color="black", linewidth=0.5)
            axes[1].set_ylabel("divergence from ours (%)")
            axes[1].set_xlabel("date")
            axes[1].legend()

        return fig

    def __repr__(self) -> str:
        return f"ConcordanceReport(engines={self._engines}, max_divergence={self.max_divergence:.4f}%)"
