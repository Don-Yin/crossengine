"""Portfolio state -- cash, positions, and derived quantities.

The Portfolio is the single source of truth for "what does my portfolio look
like right now".  The engine mutates it via :meth:`fill`; everything else
reads from it.

Adapted from earlier PortfolioState / PortfolioEnv prototypes with cleaner
separation: this class owns *state*, the engine owns *logic*.
"""

from __future__ import annotations


class Portfolio:
    """Mutable portfolio state tracking cash and share positions.

    Parameters
    ----------
    cash : float
        Starting cash balance.
    assets : list[str]
        Universe of tradeable asset names.
    """

    __slots__ = ("_cash", "_positions", "_assets")

    def __init__(self, cash: float, assets: list[str]) -> None:
        self._cash = float(cash)
        self._assets = list(assets)
        self._positions: dict[str, float] = {a: 0.0 for a in assets}

    # -- read-only properties ------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, float]:
        """Share counts per asset (shallow copy -- safe to pass around)."""
        return dict(self._positions)

    @property
    def assets(self) -> list[str]:
        return self._assets

    def shares(self, asset: str) -> float:
        """Share count for a single asset."""
        return self._positions.get(asset, 0.0)

    # -- derived quantities (require current prices) -------------------------

    def equity(self, prices: dict[str, float]) -> float:
        """Total market value of all positions."""
        return sum(self._positions[a] * prices.get(a, 0.0) for a in self._positions)

    def total_value(self, prices: dict[str, float]) -> float:
        """Cash + equity."""
        return self._cash + self.equity(prices)

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        """Per-asset weight = position_value / total_value."""
        tv = self.total_value(prices)
        if tv <= 0:
            return {a: 0.0 for a in self._positions}
        return {a: self._positions[a] * prices.get(a, 0.0) / tv for a in self._positions}

    # -- mutations -----------------------------------------------------------

    def fill(
        self,
        asset: str,
        side: str,
        qty: float,
        exec_price: float,
        commission: float,
    ) -> None:
        """Apply a single fill.  Mutates cash and position in-place."""
        if side == "buy":
            self._cash -= qty * exec_price + commission
            self._positions[asset] += qty
        else:
            self._cash += qty * exec_price - commission
            self._positions[asset] -= qty

    # -- serialisation -------------------------------------------------------

    def snapshot(self, prices: dict[str, float], date: object) -> dict:
        """Build a chronicle row capturing the full state at *date*."""
        tv = self.total_value(prices)
        record: dict = {"date": date, "cash": self._cash, "total_value": tv}
        for a in self._positions:
            val = self._positions[a] * prices.get(a, 0.0)
            record[f"w:{a}"] = val / tv if tv > 0 else 0.0
            record[f"pos:{a}"] = self._positions[a]
        return record

    def __repr__(self) -> str:
        held = {a: round(s, 4) for a, s in self._positions.items() if abs(s) > 1e-10}
        return f"Portfolio(cash={self._cash:.2f}, positions={held})"
