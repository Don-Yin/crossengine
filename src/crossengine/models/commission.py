"""Commission models.

Every model exposes a single method::

    def compute(self, quantity: float, price: float) -> float

``quantity`` is always positive (absolute number of shares).

Pass a bare ``float`` wherever a commission is expected and the engine will
wrap it in :class:`FlatRate` automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class CommissionModel(Protocol):
    def compute(self, quantity: float, price: float) -> float: ...


@dataclass
class FlatRate:
    """Proportional fee on trade value."""

    rate: float = 0.001
    min_fee: float = 0.0

    def compute(self, quantity: float, price: float) -> float:
        return max(abs(quantity) * price * self.rate, self.min_fee)


@dataclass
class IBKRTiered:
    """Interactive Brokers Pro -- Tiered pricing for US equities."""

    per_share: float = 0.0035
    min_per_trade: float = 0.35
    max_pct: float = 0.01

    def compute(self, quantity: float, price: float) -> float:
        trade_value = abs(quantity) * price
        cost = abs(quantity) * self.per_share
        cost = max(cost, self.min_per_trade)
        return min(cost, trade_value * self.max_pct)


@dataclass
class IBKRFixed:
    """Interactive Brokers Pro -- Fixed pricing for US equities."""

    per_share: float = 0.005
    min_per_trade: float = 1.00
    max_pct: float = 0.01

    def compute(self, quantity: float, price: float) -> float:
        trade_value = abs(quantity) * price
        cost = abs(quantity) * self.per_share
        cost = max(cost, self.min_per_trade)
        return min(cost, trade_value * self.max_pct)


@dataclass
class NoCommission:
    def compute(self, quantity: float, price: float) -> float:
        return 0.0


def make_commission(spec: float | CommissionModel) -> CommissionModel:
    """Coerce a scalar or model into a :class:`CommissionModel`."""
    if isinstance(spec, (int, float)):
        return FlatRate(rate=float(spec))
    if isinstance(spec, CommissionModel):
        return spec
    raise TypeError(f"commission must be a float or CommissionModel, got {type(spec)}")
