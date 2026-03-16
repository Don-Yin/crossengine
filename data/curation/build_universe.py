"""build the expanded 180-stock universe.json from verified candidates."""

from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
OUTPUT_PATH = DATA_DIR / "universe.json"

GICS_CODES = {
    "information_technology": "45",
    "communication_services": "50",
    "consumer_discretionary": "25",
    "financials": "40",
    "healthcare": "35",
    "industrials": "20",
    "consumer_staples": "30",
    "energy": "10",
    "utilities": "55",
    "real_estate": "60",
    "materials": "15",
}

MEGA_CAP = {
    "AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "META", "AMZN", "TSLA",
    "JPM", "JNJ", "UNH", "LLY", "PG", "WMT", "XOM",
}

STOCKS: dict[str, list[tuple[str, str]]] = {
    "information_technology": [
        ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"),
        ("AVGO", "Broadcom"), ("ORCL", "Oracle"), ("CRM", "Salesforce"),
        ("AMD", "Advanced Micro Devices"), ("ADBE", "Adobe"),
        ("ACN", "Accenture"), ("CSCO", "Cisco Systems"), ("INTC", "Intel"),
        ("QCOM", "Qualcomm"), ("TXN", "Texas Instruments"), ("INTU", "Intuit"),
        ("AMAT", "Applied Materials"), ("NOW", "ServiceNow"), ("KLAC", "KLA"),
    ],
    "communication_services": [
        ("GOOGL", "Alphabet"), ("META", "Meta Platforms"), ("NFLX", "Netflix"),
        ("DIS", "Walt Disney"), ("CMCSA", "Comcast"), ("TMUS", "T-Mobile US"),
        ("VZ", "Verizon Communications"), ("T", "AT&T"),
        ("CHTR", "Charter Communications"), ("EA", "Electronic Arts"),
        ("TTWO", "Take-Two Interactive"),
        ("OMC", "Omnicom Group"), ("LYV", "Live Nation Entertainment"),
        ("NWSA", "News Corp"), ("WBD", "Warner Bros. Discovery"),
        ("MTCH", "Match Group"),
    ],
    "consumer_discretionary": [
        ("AMZN", "Amazon"), ("TSLA", "Tesla"), ("HD", "Home Depot"),
        ("MCD", "McDonald's"), ("NKE", "Nike"), ("LOW", "Lowe's"),
        ("SBUX", "Starbucks"), ("TJX", "TJX Companies"),
        ("BKNG", "Booking Holdings"), ("MAR", "Marriott International"),
        ("CMG", "Chipotle Mexican Grill"),
        ("ORLY", "O'Reilly Automotive"), ("AZO", "AutoZone"), ("ROST", "Ross Stores"),
        ("DHI", "D.R. Horton"), ("LEN", "Lennar"), ("GM", "General Motors"),
    ],
    "financials": [
        ("JPM", "JPMorgan Chase"), ("BAC", "Bank of America"),
        ("GS", "Goldman Sachs"), ("WFC", "Wells Fargo"),
        ("MS", "Morgan Stanley"), ("BLK", "BlackRock"),
        ("SPGI", "S&P Global"), ("AXP", "American Express"),
        ("C", "Citigroup"), ("PNC", "PNC Financial Services"),
        ("USB", "U.S. Bancorp"),
        ("SCHW", "Charles Schwab"), ("COF", "Capital One Financial"),
        ("ICE", "Intercontinental Exchange"),
        ("CME", "CME Group"), ("CB", "Chubb"),
    ],
    "healthcare": [
        ("JNJ", "Johnson & Johnson"), ("UNH", "UnitedHealth Group"),
        ("PFE", "Pfizer"), ("LLY", "Eli Lilly"), ("ABT", "Abbott Laboratories"),
        ("TMO", "Thermo Fisher Scientific"), ("MRK", "Merck"),
        ("ABBV", "AbbVie"), ("DHR", "Danaher"), ("AMGN", "Amgen"),
        ("BMY", "Bristol-Myers Squibb"),
        ("ISRG", "Intuitive Surgical"), ("SYK", "Stryker"), ("MDT", "Medtronic"),
        ("CVS", "CVS Health"), ("CI", "Cigna Group"),
    ],
    "industrials": [
        ("CAT", "Caterpillar"), ("HON", "Honeywell"),
        ("UPS", "United Parcel Service"), ("RTX", "RTX Corporation"),
        ("GE", "GE Aerospace"), ("DE", "Deere & Company"),
        ("BA", "Boeing"), ("LMT", "Lockheed Martin"),
        ("UNP", "Union Pacific"), ("MMM", "3M"), ("FDX", "FedEx"),
        ("WM", "Waste Management"), ("ETN", "Eaton"), ("CSX", "CSX"),
        ("NSC", "Norfolk Southern"), ("GD", "General Dynamics"),
    ],
    "consumer_staples": [
        ("PG", "Procter & Gamble"), ("KO", "Coca-Cola"), ("PEP", "PepsiCo"),
        ("WMT", "Walmart"), ("COST", "Costco Wholesale"),
        ("PM", "Philip Morris International"), ("MO", "Altria Group"),
        ("CL", "Colgate-Palmolive"), ("KMB", "Kimberly-Clark"),
        ("GIS", "General Mills"), ("ADM", "Archer-Daniels-Midland"),
        ("MDLZ", "Mondelez International"), ("SYY", "Sysco"),
        ("HSY", "Hershey"), ("KR", "Kroger"), ("STZ", "Constellation Brands"),
    ],
    "energy": [
        ("XOM", "Exxon Mobil"), ("CVX", "Chevron"), ("COP", "ConocoPhillips"),
        ("SLB", "Schlumberger"), ("EOG", "EOG Resources"),
        ("MPC", "Marathon Petroleum"), ("PSX", "Phillips 66"),
        ("VLO", "Valero Energy"), ("OXY", "Occidental Petroleum"),
        ("WMB", "Williams Companies"), ("KMI", "Kinder Morgan"),
        ("DVN", "Devon Energy"), ("HAL", "Halliburton"),
        ("BKR", "Baker Hughes"), ("FANG", "Diamondback Energy"),
        ("CTRA", "Coterra Energy"),
    ],
    "utilities": [
        ("NEE", "NextEra Energy"), ("DUK", "Duke Energy"),
        ("SO", "Southern Company"), ("AEP", "American Electric Power"),
        ("D", "Dominion Energy"), ("EXC", "Exelon"), ("SRE", "Sempra"),
        ("XEL", "Xcel Energy"), ("WEC", "WEC Energy Group"),
        ("ES", "Eversource Energy"), ("ED", "Consolidated Edison"),
        ("PEG", "Public Service Enterprise Group"), ("AWK", "American Water Works"),
        ("CMS", "CMS Energy"), ("DTE", "DTE Energy"), ("ATO", "Atmos Energy"),
        ("PPL", "PPL Corporation"),
    ],
    "real_estate": [
        ("AMT", "American Tower"), ("PLD", "Prologis"),
        ("CCI", "Crown Castle"), ("EQIX", "Equinix"),
        ("SPG", "Simon Property Group"), ("PSA", "Public Storage"),
        ("O", "Realty Income"), ("DLR", "Digital Realty Trust"),
        ("WELL", "Welltower"), ("VICI", "VICI Properties"),
        ("AVB", "AvalonBay Communities"),
        ("EXR", "Extra Space Storage"), ("MAA", "Mid-America Apartment Communities"),
        ("UDR", "UDR"), ("REG", "Regency Centers"), ("IRM", "Iron Mountain"),
        ("SUI", "Sun Communities"),
    ],
    "materials": [
        ("LIN", "Linde"), ("APD", "Air Products & Chemicals"),
        ("SHW", "Sherwin-Williams"), ("ECL", "Ecolab"),
        ("NEM", "Newmont"), ("DD", "DuPont de Nemours"),
        ("NUE", "Nucor"), ("FCX", "Freeport-McMoRan"),
        ("VMC", "Vulcan Materials"), ("MLM", "Martin Marietta Materials"),
        ("MOS", "Mosaic"), ("PKG", "Packaging Corp of America"), ("IFF", "International Flavors & Fragrances"),
        ("PPG", "PPG Industries"), ("EMN", "Eastman Chemical"), ("CF", "CF Industries"),
    ],
}


def build_asset_entry(ticker: str, name: str, sector: str) -> dict:
    """build a single asset dict for the universe file."""
    return {
        "name": name,
        "sector": sector,
        "gics_code": GICS_CODES[sector],
        "market_cap_tier": "mega" if ticker in MEGA_CAP else "large",
    }


def main() -> None:
    assets = {}
    sector_counts = {}
    for sector, stock_list in STOCKS.items():
        sector_counts[sector] = len(stock_list)
        for ticker, name in stock_list:
            assets[ticker] = build_asset_entry(ticker, name, sector)

    total = len(assets)
    per_sector_str = ", ".join(f"{s}: {c}" for s, c in sector_counts.items())

    universe = {
        "_metadata": {
            "description": "asset universe for backtesting engine comparison study",
            "selection_frame": "S&P 500 constituents",
            "selection_rule": (
                "top ~16-17 by market capitalization within each GICS sector, "
                "requiring continuous trading history from 2018-01-01 to "
                "2025-01-01 with no gaps >5 days"
            ),
            "total_assets": total,
            "sectors": len(sector_counts),
            "per_sector": sector_counts,
            "date_range": {"start": "2020-01-01", "end": "2025-01-01"},
            "rationale": (
                "expanded to 180 stocks to support 30 non-overlapping "
                "6-stock buckets (each stock in exactly 1 bucket). the larger "
                "universe enables rerandomization-based stratification (Morgan & "
                "Rubin 2012) where compositional independence across buckets "
                "eliminates shared-stock confounds in engine comparison."
            ),
            "references": [
                "DeMiguel, Garlappi, Uppal (2009). optimal versus naive diversification. Review of Financial Studies 22(5), 1915-1953.",
                "Morgan, Rubin (2012). rerandomization to improve covariate balance in experiments. Annals of Statistics 40(2), 1263-1282.",
                "Gu, Kelly, Xiu (2020). empirical asset pricing via machine learning. Review of Financial Studies 33(5), 2223-2273.",
            ],
            "limitations": [
                "survivorship bias: only current S&P 500 members are included; stocks delisted during 2020-2025 are excluded",
                "large-cap only: S&P 500 membership implies large capitalization; engine divergence patterns on small-cap or micro-cap stocks may differ",
                "5-year window: shorter than the 30-60 year spans typical of factor studies, though sufficient for engine-comparison purposes",
            ],
        },
        "assets": assets,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(universe, f, indent=2)
        f.write("\n")

    print(f"written {total} assets to {OUTPUT_PATH}")
    print(f"sectors: {per_sector_str}")


if __name__ == "__main__":
    main()
