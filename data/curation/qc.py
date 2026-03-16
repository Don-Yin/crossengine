"""validate downloaded data quality and produce statistics + visualizations.

reads data/close.parquet (180 tickers x ~1761 trading days) and
data/universe.json (sector metadata).  outputs go to data/qc/.

usage:
    python data/curation/qc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
UNIVERSE_PATH = DATA_DIR / "universe.json"
CLOSE_PATH = DATA_DIR / "close.parquet"
QC_DIR = DATA_DIR / "qc"

TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# nature-grade rc params
#   fonts: sans-serif (Helvetica/Arial), 7 pt body, 8 pt bold titles
#   spines: top/right removed (overridden locally for heatmaps)
#   DPI: 300 for publication
#   fonttype 42: TrueType embedding in PDF
# ---------------------------------------------------------------------------
NATURE_RC: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.default": "regular",
    "axes.linewidth": 0.5,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "axes.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 0.8,
}

# colorblind-safe sector palette
# combines Wong (2011, Nature Methods) and Tol (2021) palettes
SECTOR_COLORS: dict[str, str] = {
    "information_technology": "#0072B2",
    "communication_services": "#E69F00",
    "consumer_discretionary": "#009E73",
    "financials":             "#D55E00",
    "healthcare":             "#CC79A7",
    "industrials":            "#56B4E9",
    "consumer_staples":       "#999933",
    "energy":                 "#666666",
    "utilities":              "#882255",
    "real_estate":            "#44AA99",
    "materials":              "#332288",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_universe(path: Path | None = None) -> dict:
    """load universe.json and return the full dict."""
    p = path or UNIVERSE_PATH
    if not p.exists():
        raise FileNotFoundError(f"universe.json not found at {p}")
    with open(p) as f:
        data = json.load(f)
    if "assets" not in data:
        raise ValueError("universe.json missing required 'assets' key")
    return data


def sector_order(universe: dict) -> list[str]:
    """return sectors in a fixed order derived from the universe file."""
    seen: list[str] = []
    for info in universe["assets"].values():
        s = info["sector"]
        if s not in seen:
            seen.append(s)
    return seen


def tickers_by_sector(universe: dict) -> dict[str, list[str]]:
    """return {sector: [ticker, ...]} preserving sector order."""
    out: dict[str, list[str]] = {}
    for ticker, info in universe["assets"].items():
        out.setdefault(info["sector"], []).append(ticker)
    return out


def sector_ordered_tickers(universe: dict) -> list[str]:
    """return flat ticker list ordered by sector then alphabetical."""
    tbs = tickers_by_sector(universe)
    order = sector_order(universe)
    result: list[str] = []
    for s in order:
        result.extend(sorted(tbs.get(s, [])))
    return result


def max_drawdown(prices: pd.Series) -> float:
    """compute maximum drawdown as a positive percentage.

    requires len(prices) >= 2 and all values > 0.
    returns 0.0 for degenerate inputs.
    """
    if len(prices) < 2:
        return 0.0
    cummax = prices.cummax()
    mask = cummax > 0
    dd = pd.Series(0.0, index=prices.index)
    dd[mask] = (prices[mask] - cummax[mask]) / cummax[mask]
    return float(-dd.min() * 100)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def compute_stats(close: pd.DataFrame, universe: dict) -> dict:
    """compute per-ticker and universe-level statistics."""
    if close.empty:
        raise ValueError("close DataFrame is empty; cannot compute stats")

    assets = universe["assets"]
    returns = close.pct_change().iloc[1:]

    per_ticker = {}
    for ticker in sorted(close.columns):
        series = close[ticker]
        valid = series.dropna()
        ret = returns[ticker].dropna()
        info = assets.get(ticker, {})

        per_ticker[ticker] = {
            "ticker": ticker,
            "sector": info.get("sector", ""),
            "name": info.get("name", ""),
            "start_date": str(valid.index[0].date()) if len(valid) > 0 else None,
            "end_date": str(valid.index[-1].date()) if len(valid) > 0 else None,
            "trading_days": int(series.notna().sum()),
            "missing_days": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 4),
            "mean_daily_return": round(float(ret.mean()), 6),
            "annualized_volatility": round(
                float(ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)), 4
            ),
            "total_return_pct": round(
                float((valid.iloc[-1] / valid.iloc[0] - 1) * 100), 2
            ) if len(valid) >= 2 else None,
            "max_drawdown_pct": round(max_drawdown(valid), 2) if len(valid) >= 2 else None,
            "min_price": round(float(valid.min()), 2) if len(valid) > 0 else None,
            "max_price": round(float(valid.max()), 2) if len(valid) > 0 else None,
            "final_price": round(float(valid.iloc[-1]), 2) if len(valid) > 0 else None,
        }

    corr_matrix = returns.corr()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    ).stack()

    summary = {
        "total_tickers": len(close.columns),
        "total_trading_days": len(close),
        "complete_rows": int(close.dropna().shape[0]),
        "date_range_start": str(close.index[0].date()),
        "date_range_end": str(close.index[-1].date()),
        "tickers_with_no_missing": sum(
            1 for v in per_ticker.values() if v["missing_days"] == 0
        ),
        "mean_annualized_volatility": round(
            float(np.mean([v["annualized_volatility"] for v in per_ticker.values()])), 4
        ),
        "cross_correlation_mean": round(float(upper_tri.mean()), 4),
        "cross_correlation_min": round(float(upper_tri.min()), 4),
        "cross_correlation_max": round(float(upper_tri.max()), 4),
    }

    return {"per_ticker": per_ticker, "summary": summary}


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_prices(close: pd.DataFrame, universe: dict, path: Path) -> None:
    """normalized price chart (base 100) with log y-axis, colored by sector."""
    tbs = tickers_by_sector(universe)
    sectors = sector_order(universe)
    norm = close.div(close.bfill().iloc[0]) * 100

    start_yr = close.index[0].year
    end_yr = close.index[-1].year

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")

        for sector in sectors:
            color = SECTOR_COLORS[sector]
            tickers = sorted(tbs.get(sector, []))
            for i, ticker in enumerate(tickers):
                if ticker in norm.columns:
                    label = sector.replace("_", " ") if i == 0 else None
                    ax.plot(
                        norm.index, norm[ticker],
                        color=color, linewidth=0.6, alpha=0.85, label=label,
                    )

        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_yticks([25, 50, 100, 200, 500, 1000, 2500])
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.axhline(100, color="#999999", linewidth=0.3, linestyle="--", zorder=0)

        ax.set_title(f"normalized prices (base 100, log scale), {start_yr}-{end_yr}")
        ax.set_ylabel("price (base 100)")
        ax.set_xlabel("")
        ax.legend(loc="upper left", ncol=3, borderaxespad=0.3)

        ax.tick_params(axis="x", rotation=0)
        fig.savefig(path)
        plt.close(fig)
    print(f"  saved {path.name}")


def plot_correlation_heatmap(close: pd.DataFrame, universe: dict, path: Path) -> None:
    """pairwise return correlation heatmap ordered by sector with boundary lines."""
    ordered = sector_ordered_tickers(universe)
    ordered = [t for t in ordered if t in close.columns]
    returns = close[ordered].pct_change().iloc[1:]
    corr = returns.corr()

    assets = universe["assets"]
    tbs = tickers_by_sector(universe)
    sectors = sector_order(universe)

    heatmap_rc = {**NATURE_RC, "axes.spines.top": True, "axes.spines.right": True}

    with plt.rc_context(heatmap_rc):
        fig, ax = plt.subplots(figsize=(8.0, 7.5), layout="constrained")
        im = ax.imshow(
            corr.values, cmap="RdBu_r", vmin=-0.1, vmax=0.9,
            aspect="equal", interpolation="nearest",
        )

        ax.set_xticks(range(len(ordered)))
        ax.set_yticks(range(len(ordered)))
        ax.set_xticklabels(ordered, rotation=90, fontsize=2.8)
        ax.set_yticklabels(ordered, fontsize=2.8)

        for i, ticker in enumerate(ordered):
            sector = assets.get(ticker, {}).get("sector", "")
            color = SECTOR_COLORS.get(sector, "black")
            ax.get_xticklabels()[i].set_color(color)
            ax.get_yticklabels()[i].set_color(color)

        pos = 0
        for sec in sectors:
            n_in_sector = len([t for t in ordered if assets.get(t, {}).get("sector") == sec])
            if pos > 0:
                ax.axhline(pos - 0.5, color="white", linewidth=0.8)
                ax.axvline(pos - 0.5, color="white", linewidth=0.8)
            pos += n_in_sector

        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=30)
        cbar.ax.tick_params(labelsize=5)
        cbar.set_label("correlation", fontsize=6)
        ax.set_title("pairwise return correlation (ordered by sector)")
        fig.savefig(path)
        plt.close(fig)
    print(f"  saved {path.name}")


def plot_volatility(close: pd.DataFrame, universe: dict, path: Path) -> None:
    """KDE ridge plot of annualized volatility distributions by sector."""
    import seaborn as sns

    returns = close.pct_change().iloc[1:]
    vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    assets = universe["assets"]
    sectors = sector_order(universe)

    rows = []
    for ticker, v in vol.items():
        sector = assets.get(ticker, {}).get("sector", "")
        rows.append({"sector": sector.replace("_", " "), "volatility": float(v)})
    df = pd.DataFrame(rows)

    sector_medians = df.groupby("sector")["volatility"].median().sort_values()
    sector_labels = sector_medians.index.tolist()

    pal = [SECTOR_COLORS.get(s.replace(" ", "_"), "#666") for s in sector_labels]

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), "font.size": 7})
    g = sns.FacetGrid(df, row="sector", hue="sector", aspect=12, height=0.45,
                      palette=pal, row_order=sector_labels, hue_order=sector_labels)
    g.map(sns.kdeplot, "volatility", bw_adjust=0.8, clip_on=False, fill=True, alpha=0.85, linewidth=1.0)
    g.map(sns.kdeplot, "volatility", clip_on=False, color="w", lw=1.5, bw_adjust=0.8)
    g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

    def _label(x, color, label):
        """add sector label to ridge row."""
        ax = plt.gca()
        ax.text(0, 0.15, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes, fontsize=6)

    g.map(_label, "volatility")
    g.figure.subplots_adjust(hspace=-0.3)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.axes[-1, 0].set_xlabel("annualized volatility", fontsize=7)
    g.figure.suptitle("volatility distribution by sector (KDE)", fontsize=8,
                      fontweight="bold", y=1.02)
    g.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(g.figure)
    print(f"  saved {path.name}")


def plot_missing_data(close: pd.DataFrame, universe: dict, path: Path) -> None:
    """coverage heatmap showing fraction of valid data per ticker per month."""
    ordered = sector_ordered_tickers(universe)
    ordered = [t for t in ordered if t in close.columns]
    assets = universe["assets"]

    monthly_valid = close[ordered].resample("ME").apply(lambda x: x.notna().sum())
    trading_days_per_month = close[ordered].resample("ME").size()
    coverage = monthly_valid.div(trading_days_per_month, axis=0).T

    month_labels = [d.strftime("%Y-%m") for d in coverage.columns]

    heatmap_rc = {**NATURE_RC, "axes.spines.top": True, "axes.spines.right": True}

    with plt.rc_context(heatmap_rc):
        fig, ax = plt.subplots(figsize=(7.2, 7.0), layout="constrained")

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "coverage", ["#d62728", "#f7f7f7", "#2ca02c"], N=256
        )
        im = ax.imshow(coverage.values, cmap=cmap, vmin=0, vmax=1, aspect="auto",
                       interpolation="nearest")

        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(ordered, fontsize=2.8)

        step = max(1, len(month_labels) // 18)
        ax.set_xticks(range(0, len(month_labels), step))
        ax.set_xticklabels(
            [month_labels[i] for i in range(0, len(month_labels), step)],
            rotation=45, ha="right", fontsize=5,
        )

        for i, ticker in enumerate(ordered):
            sector = assets.get(ticker, {}).get("sector", "")
            color = SECTOR_COLORS.get(sector, "black")
            ax.get_yticklabels()[i].set_color(color)

        pct_complete = float(coverage.values.mean() * 100)
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, aspect=25)
        cbar.ax.tick_params(labelsize=5)
        cbar.set_label("coverage fraction", fontsize=6)
        ax.set_title(f"data coverage by month ({pct_complete:.0f}% complete)")
        fig.savefig(path)
        plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def print_summary(stats: dict) -> None:
    """print a concise summary to stdout."""
    s = stats["summary"]
    pt = stats["per_ticker"]

    print("\n--- quality control summary ---")
    print(f"  tickers:       {s['total_tickers']}")
    print(f"  date range:    {s['date_range_start']} -> {s['date_range_end']}")
    print(f"  trading days:  {s['total_trading_days']}")
    print(f"  complete rows: {s['complete_rows']}")

    warnings = [
        (t, v["missing_pct"])
        for t, v in pt.items()
        if v["missing_pct"] > 1.0
    ]
    if warnings:
        print(f"\n  warnings ({len(warnings)} tickers with > 1% missing):")
        for t, pct in sorted(warnings, key=lambda x: -x[1]):
            print(f"    {t}: {pct:.2f}% missing")
    else:
        print("\n  no tickers with > 1% missing data")

    print(
        f"\n  cross-sector correlation range: "
        f"{s['cross_correlation_min']:.4f} to {s['cross_correlation_max']:.4f} "
        f"(mean {s['cross_correlation_mean']:.4f})"
    )
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """run all quality checks and save outputs to data/qc/."""
    if not CLOSE_PATH.exists():
        print(f"error: {CLOSE_PATH} not found; run download_yfinance.py first",
              file=sys.stderr)
        sys.exit(1)

    print("loading data ...")
    close = pd.read_parquet(CLOSE_PATH)
    universe = load_universe()
    print(f"  close.parquet: {close.shape[0]} rows x {close.shape[1]} columns")
    print(f"  universe.json: {len(universe['assets'])} assets\n")

    QC_DIR.mkdir(parents=True, exist_ok=True)

    print("computing statistics ...")
    stats = compute_stats(close, universe)
    stats_path = QC_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  saved {stats_path.name}")

    print("\ngenerating plots ...")
    plot_prices(close, universe, QC_DIR / "prices.png")
    plot_correlation_heatmap(close, universe, QC_DIR / "correlation-heatmap.png")
    plot_volatility(close, universe, QC_DIR / "volatility.png")
    plot_missing_data(close, universe, QC_DIR / "missing-data.png")

    print_summary(stats)
    print("done.")


if __name__ == "__main__":
    main()
