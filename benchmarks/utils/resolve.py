"""STAY resolution for the benchmarking framework.

canonical implementation lives in crossengine.concordance.resolve; re-exported
here for backward compatibility with existing benchmark scripts.
"""
from crossengine.concordance.resolve import has_stay, resolve_stay

__all__ = ["has_stay", "resolve_stay"]
