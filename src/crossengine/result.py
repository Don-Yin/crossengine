from __future__ import annotations

import pandas as pd

from .metrics import compute_all_metrics, compute_benchmark_metrics, compute_returns

_ASSET_PALETTE = ["#2E5090", "#E67E22", "#27AE60", "#8E44AD", "#C0392B"]

_PUB_RC = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "legend.frameon": True,
    "legend.edgecolor": "#CCCCCC",
    "legend.fancybox": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


class BacktestResult:
    """Rich container for backtest output with metrics and plotting."""

    def __init__(
        self,
        chronicle: pd.DataFrame,
        trades: pd.DataFrame,
        risk_free: float,
        benchmark: pd.Series | None = None,
    ) -> None:
        self._chronicle = chronicle
        self._trades = trades
        self._risk_free = risk_free
        self._benchmark = benchmark

    @property
    def benchmark(self) -> pd.Series | None:
        return self._benchmark

    @benchmark.setter
    def benchmark(self, value: pd.Series | None) -> None:
        self._benchmark = value

    @property
    def portfolio_value(self) -> pd.Series:
        return self._chronicle["total_value"]

    @property
    def cash(self) -> pd.Series:
        return self._chronicle["cash"]

    @property
    def trades(self) -> pd.DataFrame:
        return self._trades

    @property
    def returns(self) -> pd.Series:
        return compute_returns(self.portfolio_value)

    def weights(self) -> pd.DataFrame:
        cols = [c for c in self._chronicle.columns if c.startswith("w:")]
        df = self._chronicle[cols].copy()
        df.columns = [c.removeprefix("w:") for c in df.columns]
        return df

    def positions(self) -> pd.DataFrame:
        cols = [c for c in self._chronicle.columns if c.startswith("pos:")]
        df = self._chronicle[cols].copy()
        df.columns = [c.removeprefix("pos:") for c in df.columns]
        return df

    @property
    def metrics(self) -> dict:
        total_comm = float(self._trades["commission"].sum()) if len(self._trades) else 0.0
        total_slip = float(self._trades["slippage"].sum()) if "slippage" in self._trades.columns and len(self._trades) else 0.0
        return compute_all_metrics(
            self.portfolio_value,
            total_comm,
            total_slip,
            len(self._trades),
            self._risk_free,
        )

    @property
    def report(self) -> str:
        """Human-readable text report of all metrics."""
        m = self.metrics
        lines = [
            f"{'Metric':<30} {'Value':>15}",
            "-" * 46,
            f"{'Period':<30} {m['start_date']} to {m['end_date']:>0}",
            f"{'Trading days':<30} {m['trading_days']:>15}",
            f"{'Initial value':<30} {m['initial_value']:>15,.2f}",
            f"{'Final value':<30} {m['final_value']:>15,.2f}",
            "",
            f"{'Total return':<30} {m['total_return_pct']:>14.4f}%",
            f"{'CAGR':<30} {m['cagr_pct']:>14.4f}%",
            f"{'Ann. volatility':<30} {m['ann_volatility_pct']:>14.4f}%",
            f"{'Max drawdown':<30} {m['max_drawdown_pct']:>14.4f}%",
            f"{'Max drawdown duration':<30} {m['max_drawdown_duration']:>15}",
            "",
            f"{'Sharpe ratio':<30} {m['sharpe_ratio']:>15.4f}",
            f"{'Sortino ratio':<30} {m['sortino_ratio']:>15.4f}",
            f"{'Calmar ratio':<30} {m['calmar_ratio']:>15.4f}",
            f"{'Omega ratio':<30} {m['omega_ratio']:>15.4f}",
            "",
            f"{'VaR 95%':<30} {m['var_95_pct']:>14.4f}%",
            f"{'CVaR 95%':<30} {m['cvar_95_pct']:>14.4f}%",
            f"{'Skewness':<30} {m['skewness']:>15.4f}",
            f"{'Kurtosis':<30} {m['kurtosis']:>15.4f}",
            f"{'Win rate':<30} {m['win_rate_pct']:>14.4f}%",
            "",
            f"{'Num trades':<30} {m['num_trades']:>15}",
            f"{'Total commissions':<30} {m['total_commissions']:>15.4f}",
            f"{'Total slippage':<30} {m['total_slippage']:>15.4f}",
        ]
        if self._benchmark is not None:
            bm = compute_benchmark_metrics(
                self.portfolio_value, self._benchmark, self._risk_free,
            )
            if bm:
                lines += [
                    "",
                    f"{'--- vs benchmark ---':<30}",
                    f"{'Excess return':<30} {bm['excess_return_pct']:>14.4f}%",
                    f"{'Beta':<30} {bm['beta']:>15.4f}",
                    f"{'Alpha (ann.)':<30} {bm['alpha_ann_pct']:>14.4f}%",
                    f"{'Information ratio':<30} {bm['information_ratio']:>15.4f}",
                    f"{'Tracking error':<30} {bm['tracking_error_pct']:>14.4f}%",
                    f"{'Up capture':<30} {bm['up_capture']:>15.4f}",
                    f"{'Down capture':<30} {bm['down_capture']:>15.4f}",
                ]
        return "\n".join(lines)

    def _trade_dates(self) -> list:
        if self._trades.empty:
            return []
        return sorted(self._trades["date"].unique())

    def _asset_colors(self) -> dict[str, str]:
        """Stable color mapping: cash + each asset, consistent across subplots."""
        assets = self.weights().columns.tolist()
        colors = {"cash": "#C8C8C8"}
        for i, a in enumerate(assets):
            colors[a] = _ASSET_PALETTE[i % len(_ASSET_PALETTE)]
        return colors

    def plot(self):
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np

        with plt.rc_context(_PUB_RC):
            benchmark = self._benchmark
            n_sub = 6 if benchmark is not None else 5
            ratios = [3, 2, 1, 1, 1, 2] if benchmark is not None else [3, 2, 1, 1, 1]
            fig, axes = plt.subplots(
                n_sub, 1, figsize=(14, 3 * n_sub), sharex=True,
                gridspec_kw={"height_ratios": ratios},
                layout="constrained",
            )

            m = self.metrics
            tdates = self._trade_dates()
            tset = set(tdates)
            tv = self.portfolio_value
            cash = self.cash
            w = self.weights()
            idx = tv.index
            cmap = self._asset_colors()
            assets = w.columns.tolist()

            dollar_fmt = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")

            def _vlines(ax):
                for d in tdates:
                    ax.axvline(d, color="#E0E0E0", alpha=0.6, linewidth=0.5)

            stack_colors = [cmap["cash"]] + [cmap[a] for a in assets]
            areas = [cash.values] + [w[c].values * tv.values for c in assets]
            labels = ["cash"] + assets

            axes[0].stackplot(idx, *areas, labels=labels, colors=stack_colors, alpha=0.70)
            axes[0].plot(idx, tv.values, color="black", linewidth=1.2, label="total")
            if benchmark is not None:
                axes[0].plot(benchmark.index, benchmark.values, label="Benchmark",
                             linewidth=1, alpha=0.7, color="grey", linestyle="--")
            _vlines(axes[0])
            tv_trade = tv[tv.index.isin(tset)]
            axes[0].scatter(tv_trade.index, tv_trade.values,
                            color="#A93226", marker="D", s=12, zorder=5, label="rebalance")
            axes[0].legend(loc="upper left", ncol=4)
            axes[0].set_ylabel("Value ($)")
            axes[0].yaxis.set_major_formatter(dollar_fmt)
            summary = (
                f"Return: {m['total_return_pct']:.1f}%   "
                f"Sharpe: {m['sharpe_ratio']:.2f}   "
                f"Max DD: {m['max_drawdown_pct']:.1f}%"
            )
            axes[0].text(
                0.01, 0.96, summary, transform=axes[0].transAxes,
                fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="square,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", linewidth=0.5),
            )

            for col in assets:
                axes[1].plot(idx, w[col].values * tv.values,
                             label=col, linewidth=1, color=cmap[col])
            axes[1].plot(idx, cash.values, label="cash", linewidth=1,
                         linestyle="--", color=cmap["cash"])
            _vlines(axes[1])
            axes[1].set_ylabel("Position ($)")
            axes[1].yaxis.set_major_formatter(dollar_fmt)
            axes[1].legend(loc="upper left", ncol=3)

            if not w.empty:
                axes[2].stackplot(
                    w.index, *[w[c].values for c in assets],
                    labels=assets, colors=[cmap[a] for a in assets], alpha=0.70,
                )
                _vlines(axes[2])
                axes[2].set_ylabel("Weight (actual)")
                axes[2].set_ylim(0, 1)
                axes[2].legend(loc="upper left", ncol=3)

            if tdates and not w.empty:
                target = w.loc[w.index.isin(tset)].copy()
                if idx[-1] not in target.index:
                    target.loc[idx[-1]] = target.iloc[-1]
                target = target.sort_index()
                tx = target.index
                bottoms = np.zeros(len(tx))
                for col in assets:
                    tops = bottoms + target[col].values
                    axes[3].fill_between(tx, bottoms, tops, step="post",
                                         color=cmap[col], alpha=0.70, label=col)
                    bottoms = tops.copy()
                axes[3].set_ylabel("Weight (target)")
                axes[3].set_ylim(0, 1)
                axes[3].legend(loc="upper left", ncol=3)

            running_max = tv.cummax()
            dd = (tv - running_max) / running_max
            axes[4].fill_between(dd.index, dd.values, 0, color="#E74C3C", alpha=0.25)
            _vlines(axes[4])
            axes[4].set_ylabel("Drawdown")

            if benchmark is not None:
                bm_aligned = benchmark.reindex(idx, method="ffill").dropna()
                common = idx.intersection(bm_aligned.index)
                p_rebased = tv.loc[common] / tv.loc[common].iloc[0] * 100
                b_rebased = bm_aligned.loc[common] / bm_aligned.loc[common].iloc[0] * 100
                axes[5].plot(common, p_rebased, label="portfolio", linewidth=1.3, color="#2E5090")
                bm_label = benchmark.name if benchmark.name else "benchmark"
      272 +     axes[5].plot(common, b_rebased, label=bm_label, linewidth=1,
                             color="grey", linestyle="--", alpha=0.8)
                axes[5].fill_between(common, p_rebased, b_rebased,
                                     where=p_rebased >= b_rebased, color="#27AE60", alpha=0.10)
                axes[5].fill_between(common, p_rebased, b_rebased,
                                     where=p_rebased < b_rebased, color="#E74C3C", alpha=0.10)
                _vlines(axes[5])
                axes[5].set_ylabel("Rebased (100)")
                axes[5].legend(loc="upper left")

            axes[-1].set_xlabel("Date")
            axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            fig.autofmt_xdate(rotation=30, ha="right")

            return fig

    def __repr__(self) -> str:
        m = self.metrics
        return (
            f"BacktestResult("
            f"return={m['total_return_pct']:.1f}%, "
            f"sharpe={m['sharpe_ratio']:.2f}, "
            f"max_dd={m['max_drawdown_pct']:.1f}%, "
            f"trades={m['num_trades']})"
        )
