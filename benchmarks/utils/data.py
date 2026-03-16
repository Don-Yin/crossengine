"""Constants, data loaders, and rebalance-date helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = [
    "ROOT", "DATA_DIR", "RESULTS_ROOT", "BACKTEST_START",
    "T212_COMMISSION", "T212_SLIPPAGE",
    "load_close", "load_close_full", "load_spx", "compute_asset_avg",
    "month_starts", "month_start_dates", "every_day", "alternating_weights",
]

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
RESULTS_ROOT = ROOT / ".results"

T212_COMMISSION = 0.0015  # 15 bps -- Trading212 FX conversion fee
T212_SLIPPAGE = 0.0003    #  3 bps -- typical mega-cap bid-ask spread
BACKTEST_START = pd.Timestamp("2020-01-01")


def load_close() -> pd.DataFrame:
    """backtest-period close prices (2020+), all assets, no nulls."""
    df = pd.read_parquet(DATA_DIR / "close.parquet")
    return df.loc[df.index >= BACKTEST_START].dropna()


def load_close_full() -> pd.DataFrame:
    """full-history close prices (all years, no global dropna -- for ML training)."""
    return pd.read_parquet(DATA_DIR / "close.parquet")


def load_spx() -> pd.Series:
    """S&P 500 daily close as a Series (DatetimeIndex)."""
    return pd.read_parquet(DATA_DIR / "spx.parquet")["SPX"].dropna()


def compute_asset_avg(close: pd.DataFrame, initial_cash: float) -> pd.Series:
    """Equal-weight buy-and-hold of all assets in *close*, starting at *initial_cash*."""
    return (close / close.iloc[0]).mean(axis=1) * initial_cash


# ── rebalance-date helpers ────────────────────────────────────────────────

def month_starts(close: pd.DataFrame) -> set[pd.Timestamp]:
    """First trading day of each calendar month."""
    return set(
        close.groupby(close.index.to_period("M"))
        .apply(lambda g: g.index[0])
        .values
    )


month_start_dates = month_starts  # backward-compat alias


def every_day(close: pd.DataFrame) -> set[pd.Timestamp]:
    """Every trading day in the index."""
    return set(close.index)


def alternating_weights(
    tickers: list[str], rebal_dates,
) -> dict[pd.Timestamp, dict[str, float]]:
    """Return {date: {asset: weight}} alternating 80/5 <-> 5/80 splits."""
    toggle = True
    out: dict[pd.Timestamp, dict[str, float]] = {}
    for d in sorted(rebal_dates):
        if toggle:
            w = {tickers[0]: 0.8}
            for t in tickers[1:]:
                w[t] = 0.05
        else:
            w = {tickers[-1]: 0.8}
            for t in tickers[:-1]:
                w[t] = 0.05
        toggle = not toggle
        out[d] = w
    return out
