from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class OHLCV:
    """Immutable multi-asset OHLCV data container.

    Each DataFrame has a DatetimeIndex (rows = dates) and one column per asset.
    Only ``close`` is mandatory; the rest default to ``None`` and are only
    needed when the backtest requires them (e.g. ``high``/``low`` for
    limit / stop-order fill simulation).
    """

    close: pd.DataFrame
    open: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    volume: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        close = self.close
        if not isinstance(close.index, pd.DatetimeIndex):
            close = close.copy()
            close.index = pd.to_datetime(close.index)
            object.__setattr__(self, "close", close)

        for name in ("open", "high", "low", "volume"):
            df = getattr(self, name)
            if df is None:
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df.index = pd.to_datetime(df.index)
                object.__setattr__(self, name, df)
            if list(df.columns) != list(close.columns):
                raise ValueError(f"{name} columns {list(df.columns)} must match close columns {list(close.columns)}")
            if len(df) != len(close):
                raise ValueError(f"{name} has {len(df)} rows but close has {len(close)} rows")

    @property
    def assets(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def __len__(self) -> int:
        return len(self.close)

    @classmethod
    def from_long(
        cls,
        df: pd.DataFrame,
        date_col: str = "date",
        asset_col: str = "asset",
        close_col: str = "close",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        volume_col: str = "volume",
    ) -> OHLCV:
        """Build from a long-format DataFrame (one row per date × asset)."""
        close = df.pivot(index=date_col, columns=asset_col, values=close_col)
        close.index = pd.to_datetime(close.index)
        close.columns.name = None

        kwargs: dict = {"close": close}
        for field, col in [("open", open_col), ("high", high_col), ("low", low_col), ("volume", volume_col)]:
            if col in df.columns:
                pivoted = df.pivot(index=date_col, columns=asset_col, values=col)
                pivoted.index = pd.to_datetime(pivoted.index)
                pivoted.columns.name = None
                kwargs[field] = pivoted

        return cls(**kwargs)
