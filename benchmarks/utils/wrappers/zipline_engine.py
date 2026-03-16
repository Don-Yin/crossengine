"""adapter for zipline-reloaded backtesting engine (excluded).

status: excluded due to unfixable framework bug (verified March 2026).

the bug: zipline's ``calendar_utils.get_calendar`` wrapper calls
``exchange_calendars.get_calendar`` with inconsistent kwargs across
code paths. ingestion passes ``start=1990-01-01``; runtime does not.
since ``exchange_calendars`` caches by (name + kwargs), mismatched
kwargs produce distinct calendar instances that defeat the cache,
causing session-alignment assertion failures during data loading.
this is NOT an ``is`` vs ``==`` identity check issue; the actual
mechanism is kwargs mismatch in the calendar resolution architecture.
cannot be fixed by user-side configuration.

source files: ``zipline/utils/calendar_utils.py``,
``exchange_calendars/calendar_utils.py``,
``zipline/data/bundles/csvdir.py``,
``zipline/finance/trading.py``.

uses a temporary csvdir bundle to ingest close prices, then runs a
WeightSchedule-based algorithm via run_algorithm(). the wrapper
catches all exceptions and returns NaN to allow batch runs to
continue; this is necessary because the framework bug surfaces
non-deterministically depending on calendar cache state.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pandas as pd
from utils.types import WeightSchedule


def run_zipline_engine(
    close: pd.DataFrame,
    ws: WeightSchedule,
    *,
    initial_cash: float,
    commission: float,
) -> pd.Series:
    """run zipline-reloaded via csvdir bundle, return portfolio value series."""
    try:
        return _run(close, ws, initial_cash=initial_cash, commission=commission)
    except Exception as exc:
        print(f"  [zipline] engine failed: {type(exc).__name__}: {exc}")
        return pd.Series(float("nan"), index=close.index)


def _run(
    close: pd.DataFrame,
    ws: WeightSchedule,
    *,
    initial_cash: float,
    commission: float,
) -> pd.Series:
    """build csvdir bundle from close prices and execute the algorithm."""
    from exchange_calendars import get_calendar
    from zipline import run_algorithm
    from zipline.api import order_target_percent, set_commission, set_slippage, symbol
    from zipline.data.bundles import ingest, register
    from zipline.data.bundles.csvdir import csvdir_equities
    from zipline.finance.commission import PerTrade
    from zipline.finance.slippage import FixedBasisPointsSlippage

    tickers = close.columns.tolist()
    start, end = close.index[0], close.index[-1]
    bundle_name = f"_wsb_{uuid.uuid4().hex[:8]}"
    tmpdir = _write_csvdir(close, tickers)

    try:
        register(bundle_name, csvdir_equities(["daily"], tmpdir), calendar_name="NYSE")
        ingest(bundle_name)

        ws_ref = dict(ws)
        bps = commission * 10_000

        def initialize(context):
            set_slippage(FixedBasisPointsSlippage(basis_points=bps, volume_limit=1.0))
            set_commission(PerTrade(cost=0.0))
            context.ws = ws_ref
            context.sids = {t: symbol(t) for t in tickers}

        def handle_data(context, data):
            dt = pd.Timestamp(data.current_dt.date())
            weights = context.ws.get(dt)
            if weights is None:
                return
            for ticker, sid in context.sids.items():
                order_target_percent(sid, weights.get(ticker, 0.0))

        result = run_algorithm(
            start=start,
            end=end,
            initialize=initialize,
            handle_data=handle_data,
            capital_base=initial_cash,
            bundle=bundle_name,
            trading_calendar=get_calendar("NYSE"),
        )
        return result["portfolio_value"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _cleanup_bundle(bundle_name)


def _write_csvdir(close: pd.DataFrame, tickers: list[str]) -> str:
    """write per-ticker CSVs in zipline csvdir format to a temp directory."""
    tmpdir = tempfile.mkdtemp(prefix="zipline_")
    daily_dir = Path(tmpdir) / "daily"
    daily_dir.mkdir()
    for ticker in tickers:
        series = close[ticker]
        pd.DataFrame({
            "date": series.index.strftime("%Y-%m-%d"),
            "open": series.values, "high": series.values,
            "low": series.values, "close": series.values,
            "volume": 0, "dividend": 0.0, "split": 1.0,
        }).to_csv(daily_dir / f"{ticker}.csv", index=False)
    return tmpdir


def _cleanup_bundle(name: str) -> None:
    """remove ingested bundle data from the zipline root directory."""
    zipline_root = Path(os.environ.get("ZIPLINE_ROOT", str(Path.home() / ".zipline")))
    bundle_dir = zipline_root / "data" / name
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir, ignore_errors=True)
