"""verify yfinance data availability for candidate ~180 stock universe."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

START = "2020-01-01"
END = "2025-01-01"
MIN_TRADING_DAYS = 1250
MAX_NAN_GAP = 5

CANDIDATES: dict[str, list[str]] = {
    "information_technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "ACN", "CSCO", "INTC",
        "QCOM", "TXN", "INTU", "AMAT", "NOW", "KLAC", "SNPS", "CDNS", "FTNT",
    ],
    "communication_services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR", "EA", "TTWO",
        "OMC", "IPG", "LYV", "FOXA", "WBD", "MTCH", "NWSA",
    ],
    "consumer_discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "MAR", "CMG",
        "ORLY", "AZO", "ROST", "DHI", "LEN", "GM", "F", "EBAY", "POOL",
    ],
    "financials": [
        "JPM", "BAC", "GS", "WFC", "MS", "BLK", "SPGI", "AXP", "C", "PNC", "USB",
        "SCHW", "COF", "ICE", "CME", "CB", "MET", "AIG", "PRU", "ALL",
    ],
    "healthcare": [
        "JNJ", "UNH", "PFE", "LLY", "ABT", "TMO", "MRK", "ABBV", "DHR", "AMGN", "BMY",
        "ISRG", "SYK", "MDT", "CVS", "CI", "BSX", "GILD", "ZTS",
    ],
    "industrials": [
        "CAT", "HON", "UPS", "RTX", "GE", "DE", "BA", "LMT", "UNP", "MMM", "FDX",
        "WM", "ETN", "CSX", "NSC", "GD", "ITW", "EMR", "PCAR",
    ],
    "consumer_staples": [
        "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "CL", "KMB", "GIS", "ADM",
        "MDLZ", "SYY", "HSY", "KR", "STZ", "CHD", "SJM",
    ],
    "energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "WMB", "KMI",
        "DVN", "HAL", "BKR", "FANG", "CTRA", "HES", "TRGP", "OKE", "LNG",
    ],
    "utilities": [
        "NEE", "DUK", "SO", "AEP", "D", "EXC", "SRE", "XEL", "WEC", "ES", "ED",
        "PEG", "AWK", "CMS", "DTE", "ATO", "PPL", "FE", "EVRG", "CNP", "NI",
    ],
    "real_estate": [
        "AMT", "PLD", "CCI", "EQIX", "SPG", "PSA", "O", "DLR", "WELL", "VICI", "AVB",
        "EXR", "MAA", "UDR", "REG", "IRM", "SUI", "CPT", "HST", "KIM", "BXP",
    ],
    "materials": [
        "LIN", "APD", "SHW", "ECL", "NEM", "DD", "NUE", "FCX", "VMC", "MLM", "DOW",
        "CTVA", "IFF", "PPG", "EMN", "CF", "MOS", "PKG", "IP", "BLL",
    ],
}


def _max_nan_gap(series: pd.Series) -> int:
    """compute longest consecutive NaN streak in a series."""
    is_nan = series.isna()
    if not is_nan.any():
        return 0
    groups = (is_nan != is_nan.shift()).cumsum()
    return int(is_nan.groupby(groups).sum().max())


def verify_ticker(ticker: str, close: pd.Series) -> dict:
    """check a single ticker for data completeness."""
    n_days = int(close.notna().sum())
    gap = _max_nan_gap(close)
    passed = n_days >= MIN_TRADING_DAYS and gap <= MAX_NAN_GAP
    return {
        "ticker": ticker,
        "trading_days": n_days,
        "max_nan_gap": gap,
        "passed": passed,
        "reason": "" if passed else f"days={n_days}, gap={gap}",
    }


def main() -> None:
    all_tickers = sorted({t for tl in CANDIDATES.values() for t in tl})
    print(f"verifying {len(all_tickers)} candidate tickers from {START} to {END}")
    print(f"minimum trading days: {MIN_TRADING_DAYS}, max NaN gap: {MAX_NAN_GAP}\n")

    print("downloading data (this may take a minute) ...")
    raw = yf.download(all_tickers, start=START, end=END, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]

    results_by_sector: dict[str, list[dict]] = {}
    for sector, tickers in CANDIDATES.items():
        sector_results = []
        for t in tickers:
            if t not in close.columns:
                sector_results.append({
                    "ticker": t, "trading_days": 0, "max_nan_gap": 9999,
                    "passed": False, "reason": "not found in download",
                })
                continue
            sector_results.append(verify_ticker(t, close[t]))
        results_by_sector[sector] = sector_results

    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)

    selected: dict[str, list[str]] = {}
    substitutions: list[str] = []

    for sector, results in results_by_sector.items():
        passed = [r for r in results if r["passed"]]
        failed = [r for r in results if not r["passed"]]
        top_11 = [r["ticker"] for r in passed[:11]]
        selected[sector] = top_11

        print(f"\n{sector} ({len(passed)} passed, {len(failed)} failed, selected {len(top_11)}):")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            sel = " [SELECTED]" if r["ticker"] in top_11 else ""
            print(f"  {r['ticker']:6s}  {status}  days={r['trading_days']:4d}  gap={r['max_nan_gap']:3d}{sel}")
            if not r["passed"]:
                substitutions.append(f"  {r['ticker']} ({sector}): {r['reason']}")

    total = sum(len(v) for v in selected.values())
    print(f"\n{'=' * 70}")
    print(f"TOTAL SELECTED: {total} stocks across {len(selected)} sectors")
    print(f"{'=' * 70}")
    for sector, tickers in selected.items():
        print(f"  {sector}: {len(tickers)} -> {', '.join(tickers)}")

    if substitutions:
        print(f"\nFAILED TICKERS:")
        for s in substitutions:
            print(s)

    output = {"selected": selected, "total": total}
    out_path = Path(__file__).resolve().parent / "verify_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
