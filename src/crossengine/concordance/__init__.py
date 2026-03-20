"""cross-engine concordance testing for backtesting engines.

run the same strategy through multiple backtesting engines and measure
how much they disagree. the main entry point is :func:`concordance`.
"""

from crossengine.concordance.api import concordance
from crossengine.concordance.report import ConcordanceReport
from crossengine.concordance.types import STAY, SignalSchedule, WeightSchedule

__all__ = [
    "concordance",
    "ConcordanceReport",
    "STAY",
    "SignalSchedule",
    "WeightSchedule",
]
