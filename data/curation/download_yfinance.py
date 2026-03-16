"""download OHLCV data from yfinance for the full 180-stock universe.

the universe is defined in data/universe.json (16-17 stocks per GICS sector,
11 sectors, selected by market capitalisation from S&P 500
constituents with continuous history 2018-2025).

7 years of data are pulled: 2018-2025. the first 2 years (2018-2019) serve
as a dedicated ML training buffer; all strategies backtest on 2020-2025.

output goes to data/ (gitignored). each field is a separate wide-format
parquet with DatetimeIndex (rows = trading days) and one column per ticker:

    data/close.parquet   data/open.parquet   data/high.parquet
    data/low.parquet     data/volume.parquet  data/spx.parquet

usage:
    python data/curation/download_yfinance.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent                       # data/
UNIVERSE_PATH = DATA_DIR / "universe.json"

START = "2018-01-01"
END = "2025-01-01"

FIELDS = {
    "Close": "close",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Volume": "volume",
}


def _load_tickers() -> list[str]:
    """read ticker list from universe.json (canonical source of truth)."""
    with open(UNIVERSE_PATH) as f:
        universe = json.load(f)
    return sorted(universe["assets"].keys())


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    tickers = _load_tickers()

    print(f"universe: {len(tickers)} tickers from {UNIVERSE_PATH.name}")
    print(f"date range: {START} to {END}")
    print(f"tickers: {', '.join(tickers)}\n")

    print("downloading equity data ...")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        for yf_name, our_name in FIELDS.items():
            df = raw[yf_name][tickers].copy()
            df.index.name = "date"
            df.columns.name = None
            df.to_parquet(DATA_DIR / f"{our_name}.parquet")
            non_null = df.notna().all(axis=1).sum()
            print(f"  {our_name}.parquet  {len(df)} rows x {len(df.columns)} cols  ({non_null} complete)")
    else:
        ticker = tickers[0]
        for yf_name, our_name in FIELDS.items():
            df = raw[[yf_name]].copy()
            df.columns = [ticker]
            df.index.name = "date"
            df.to_parquet(DATA_DIR / f"{our_name}.parquet")
            print(f"  {our_name}.parquet  {len(df)} rows x 1 col")

    print(f"\ndownloading ^GSPC (S&P 500) ...")
    spx_raw = yf.download("^GSPC", start=START, end=END, auto_adjust=True)
    spx_close = spx_raw["Close"]
    if isinstance(spx_close, pd.DataFrame):
        spx_close = spx_close.iloc[:, 0]
    spx_close = spx_close.to_frame("SPX")
    spx_close.index.name = "date"
    spx_close.to_parquet(DATA_DIR / "spx.parquet")
    print(f"  spx.parquet  {len(spx_close)} rows")

    print(f"\ndone. data saved to {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
