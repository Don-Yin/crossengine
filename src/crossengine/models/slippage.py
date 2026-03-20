"""Slippage models.

A slippage model adjusts the execution price away from the bar's close to
simulate market impact.  ``quantity > 0`` means a buy (price worsens upward);
``quantity < 0`` means a sell (price worsens downward).

Pass a bare ``float`` wherever slippage is expected and the engine will wrap
it in :class:`FixedSlippage` automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class SlippageModel(Protocol):
    def apply(self, price: float, quantity: float, volume: float | None = None) -> float: ...


@dataclass
class FixedSlippage:
    """Constant proportional slippage (e.g. 0.001 = 10 bps)."""

    rate: float = 0.0

    def apply(self, price: float, quantity: float, volume: float | None = None) -> float:
        if quantity > 0:
            return price * (1.0 + self.rate)
        if quantity < 0:
            return price * (1.0 - self.rate)
        return price


@dataclass
class VolumeImpact:
    """Fixed rate *plus* a component proportional to ``|quantity| / volume``."""

    fixed_rate: float = 0.0
    impact_factor: float = 0.1

    def apply(self, price: float, quantity: float, volume: float | None = None) -> float:
        slip = price * self.fixed_rate
        if volume and volume > 0 and self.impact_factor > 0:
            slip += price * self.impact_factor * (abs(quantity) / volume)
        if quantity > 0:
            return price + slip
        if quantity < 0:
            return price - slip
        return price


# TODO(sqrt_impact): add SquareRootImpact model (Almgren-Chriss family).
# impact = coeff * volatility * sqrt(|quantity| / volume)
# where volatility is daily std of returns over a lookback window.
# The engine needs to pre-compute rolling volatility and pass it to apply(),
# either via an extra kwarg or by extending the SlippageModel protocol with
# an optional volatility parameter.


@dataclass
class NoSlippage:
    def apply(self, price: float, quantity: float, volume: float | None = None) -> float:
        return price


def make_slippage(spec: float | SlippageModel) -> SlippageModel:
    """Coerce a scalar or model into a :class:`SlippageModel`."""
    if isinstance(spec, (int, float)):
        return FixedSlippage(rate=float(spec))
    if isinstance(spec, SlippageModel):
        return spec
    raise TypeError(f"slippage must be a float or SlippageModel, got {type(spec)}")
