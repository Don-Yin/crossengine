"""Limit / stop order management.

Orders live in a pending queue and are checked against each bar's high / low
to determine whether a fill occurs.  The engine processes them *before* the
bar's signal-driven rebalancing so that fills are reflected in the portfolio
state when signal resolution runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import pandas as pd


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    asset: str
    side: Literal["buy", "sell"]
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    submit_date: pd.Timestamp | None = None
    valid_until: pd.Timestamp | None = None
    tag: str = ""

    _triggered: bool = field(default=False, repr=False, compare=False)


class OrderQueue:
    """Sparse collection of non-market orders submitted by the user."""

    def __init__(self, orders: list[Order] | None = None):
        self._orders: list[Order] = list(orders) if orders else []

    def add(self, order: Order) -> OrderQueue:
        self._orders.append(order)
        return self

    def on_date(self, date: pd.Timestamp) -> list[Order]:
        return [o for o in self._orders if o.submit_date is not None and o.submit_date == date]

    def __len__(self) -> int:
        return len(self._orders)


def check_pending_fill(
    order: Order,
    bar_high: float | None,
    bar_low: float | None,
    bar_close: float,
    current_date: pd.Timestamp,
) -> float | None:
    """Return the execution price if *order* fills on this bar, else ``None``.

    Uses the bar's high / low range to decide whether the trigger / limit
    price was reached during the session.

    Limit buy  : fills if bar_low  ≤ limit_price  → exec at limit (or better).
    Limit sell : fills if bar_high ≥ limit_price  → exec at limit (or better).
    Stop sell  : fills if bar_low  ≤ stop_price   → exec at stop  (or worse on gap-down).
    Stop buy   : fills if bar_high ≥ stop_price   → exec at stop  (or worse on gap-up).
    Stop-limit : stop triggers first, then limit must be reachable in the same
                 or a subsequent bar.
    """
    if order.valid_until is not None and current_date > order.valid_until:
        return None

    high = bar_high if bar_high is not None else bar_close
    low = bar_low if bar_low is not None else bar_close

    match order.order_type:
        case OrderType.LIMIT:
            lp = order.limit_price
            if lp is None:
                return None
            if order.side == "buy" and low <= lp:
                return min(lp, high)
            if order.side == "sell" and high >= lp:
                return max(lp, low)

        case OrderType.STOP:
            sp = order.stop_price
            if sp is None:
                return None
            if order.side == "sell" and low <= sp:
                return min(sp, high) if high >= sp else high
            if order.side == "buy" and high >= sp:
                return max(sp, low) if low <= sp else low

        case OrderType.STOP_LIMIT:
            sp, lp = order.stop_price, order.limit_price
            if sp is None or lp is None:
                return None
            if not order._triggered:
                if order.side == "sell" and low <= sp:
                    order._triggered = True
                elif order.side == "buy" and high >= sp:
                    order._triggered = True
            if order._triggered:
                if order.side == "buy" and low <= lp:
                    return min(lp, high)
                if order.side == "sell" and high >= lp:
                    return max(lp, low)

    return None
