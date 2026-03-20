"""shared type aliases for the benchmarking framework.

canonical definitions live in crossengine.concordance.types; re-exported here
for backward compatibility with existing benchmark scripts.
"""
from crossengine.concordance.types import STAY, SignalSchedule, WeightSchedule

__all__ = ["STAY", "SignalSchedule", "WeightSchedule"]
