"""summary.figures -- cross-benchmark publication figures (package re-exports)."""

from summary.figures._divergence import (
    plot_divergence_landscape,
    plot_divergence_timeseries,
    plot_divergence_vs_cost,
)
from summary.figures._equity import (
    plot_drawdown_comparison,
    plot_equity_overlay,
)
from summary.figures._heatmap import (
    plot_divergence_correlation,
    plot_engine_agreement,
    plot_engine_concordance,
    plot_performance_heatmap,
    plot_risk_return,
)
from summary.figures._implementation_risk import (
    plot_complexity_analysis,
    plot_divergence_anatomy,
    plot_economic_significance,
    plot_mechanism_comparison,
    plot_metric_sensitivity,
)
from summary.figures._profiling import plot_profiling_distributions
from summary.figures._statistical import (
    plot_permutation_test,
    plot_power_analysis,
    plot_qq_normality,
    plot_tost_equivalence,
    plot_wilcoxon_robustness,
)
from summary.figures._tables import (
    plot_controlled_distributions,
    plot_table_costs,
    plot_table_raw,
)

__all__ = [
    "plot_divergence_landscape",
    "plot_divergence_vs_cost",
    "plot_divergence_timeseries",
    "plot_risk_return",
    "plot_performance_heatmap",
    "plot_divergence_correlation",
    "plot_equity_overlay",
    "plot_drawdown_comparison",
    "plot_engine_concordance",
    "plot_engine_agreement",
    "plot_table_raw",
    "plot_controlled_distributions",
    "plot_table_costs",
    "plot_metric_sensitivity",
    "plot_economic_significance",
    "plot_complexity_analysis",
    "plot_divergence_anatomy",
    "plot_mechanism_comparison",
    "plot_profiling_distributions",
    "plot_tost_equivalence",
    "plot_power_analysis",
    "plot_wilcoxon_robustness",
    "plot_permutation_test",
    "plot_qq_normality",
]
