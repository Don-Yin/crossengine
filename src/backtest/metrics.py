"""Pure metric-computation functions.

All helpers accept pandas Series / DataFrames and return scalars.
"""

from __future__ import annotations

import math
from datetime import timedelta

import pandas as pd


def compute_returns(values: pd.Series) -> pd.Series:
    r = values.pct_change()
    r = r.replace([float("inf"), float("-inf")], float("nan"))
    return r.dropna()


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.04) -> float:
    if len(returns) < 2:
        return 0.0
    daily_rf = risk_free / 252
    excess = returns - daily_rf
    std = returns.std()
    if std < 1e-10:
        return 0.0
    return float(excess.mean() / std * math.sqrt(252))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.04) -> float:
    import numpy as np

    if len(returns) < 2:
        return 0.0
    daily_rf = risk_free / 252
    excess = returns - daily_rf
    downside_diff = np.minimum(excess.values, 0.0)
    down_dev = math.sqrt(float(np.mean(downside_diff ** 2)))
    if down_dev < 1e-10:
        return 0.0
    return float(excess.mean() / down_dev * math.sqrt(252))


def max_drawdown(values: pd.Series) -> tuple[float, timedelta]:
    """Return (max_drawdown_fraction, duration_of_worst_drawdown)."""
    if len(values) < 2:
        return 0.0, timedelta(0)

    running_max = values.cummax()
    dd = (values - running_max) / running_max

    worst_dd = float(dd.min())
    if worst_dd >= 0:
        return 0.0, timedelta(0)

    peak_idx = 0
    current_worst = 0.0
    worst_duration = timedelta(0)
    for i in range(len(values)):
        if values.iloc[i] >= running_max.iloc[i]:
            peak_idx = i
        if dd.iloc[i] < current_worst:
            current_worst = dd.iloc[i]
            worst_duration = values.index[i] - values.index[peak_idx]

    return worst_dd, worst_duration


def calmar_ratio(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    total_return = values.iloc[-1] / values.iloc[0] - 1
    n_years = (values.index[-1] - values.index[0]).days / 365.25
    if n_years <= 0:
        return 0.0
    base = 1 + total_return
    if base <= 0:
        return 0.0
    cagr = base ** (1 / max(n_years, 1 / 365)) - 1
    dd, _ = max_drawdown(values)
    if abs(dd) < 1e-10:
        return float("inf") if cagr > 0 else 0.0
    return float(cagr / abs(dd))


def omega_ratio(returns: pd.Series, risk_free: float = 0.04) -> float:
    if len(returns) < 2:
        return 0.0
    daily_rf = risk_free / 252
    excess = returns - daily_rf
    gains = float(excess[excess > 0].sum())
    losses = float(abs(excess[excess < 0].sum()))
    if losses < 1e-10:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def compute_benchmark_metrics(
    portfolio_values: pd.Series,
    benchmark_values: pd.Series,
    risk_free: float = 0.04,
) -> dict:
    """Metrics that describe portfolio performance relative to a benchmark."""
    pr = compute_returns(portfolio_values)
    br = compute_returns(benchmark_values)
    common = pr.index.intersection(br.index)
    pr, br = pr.loc[common], br.loc[common]
    if len(pr) < 2:
        return {}

    daily_rf = risk_free / 252
    excess = pr - br

    # beta / alpha (CAPM regression)
    cov = pr.cov(br)
    var_b = br.var()
    beta = float(cov / var_b) if var_b > 1e-14 else 0.0
    alpha_daily = float(pr.mean() - daily_rf - beta * (br.mean() - daily_rf))
    alpha_ann = alpha_daily * 252

    # tracking error / information ratio
    te = float(excess.std() * math.sqrt(252))
    ir = float(excess.mean() / excess.std() * math.sqrt(252)) if excess.std() > 1e-14 else 0.0

    # up / down capture
    up_days = br > 0
    dn_days = br < 0
    up_capture = float(pr[up_days].mean() / br[up_days].mean()) if up_days.sum() > 0 else 0.0
    dn_capture = float(pr[dn_days].mean() / br[dn_days].mean()) if dn_days.sum() > 0 else 0.0

    bm_aligned = benchmark_values.reindex(portfolio_values.index, method="ffill").dropna()
    pv_common = portfolio_values.loc[bm_aligned.index]
    p_total = pv_common.iloc[-1] / pv_common.iloc[0] - 1
    b_total = bm_aligned.iloc[-1] / bm_aligned.iloc[0] - 1

    return {
        "excess_return_pct": round((p_total - b_total) * 100, 4),
        "beta": round(beta, 4),
        "alpha_ann_pct": round(alpha_ann * 100, 4),
        "information_ratio": round(ir, 4),
        "tracking_error_pct": round(te * 100, 4),
        "up_capture": round(up_capture, 4),
        "down_capture": round(dn_capture, 4),
    }


def compute_all_metrics(
    portfolio_values: pd.Series,
    total_commissions: float,
    total_slippage: float,
    num_trades: int,
    risk_free: float = 0.04,
) -> dict:
    returns = compute_returns(portfolio_values)
    dd_pct, dd_dur = max_drawdown(portfolio_values)

    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    n_years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / max(n_years, 1 / 365)) - 1 if n_years > 0 else 0.0

    ann_vol = float(returns.std() * math.sqrt(252) * 100) if len(returns) > 1 else 0.0
    skew = float(returns.skew()) if len(returns) > 2 else 0.0
    kurt = float(returns.kurtosis()) if len(returns) > 3 else 0.0
    q05 = float(returns.quantile(0.05) * 100) if len(returns) > 20 else 0.0
    tail = returns[returns <= returns.quantile(0.05)] if len(returns) > 20 else returns
    cvar = float(tail.mean() * 100) if len(tail) > 0 else 0.0
    win = float((returns > 0).sum() / len(returns) * 100) if len(returns) > 0 else 0.0

    return {
        "start_date": str(portfolio_values.index[0].date()),
        "end_date": str(portfolio_values.index[-1].date()),
        "trading_days": len(portfolio_values),
        "total_return_pct": round(total_return * 100, 4),
        "cagr_pct": round(cagr * 100, 4),
        "ann_volatility_pct": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe_ratio(returns, risk_free), 4),
        "sortino_ratio": round(sortino_ratio(returns, risk_free), 4),
        "calmar_ratio": round(calmar_ratio(portfolio_values), 4),
        "omega_ratio": round(omega_ratio(returns, risk_free), 4),
        "max_drawdown_pct": round(dd_pct * 100, 4),
        "max_drawdown_duration": str(dd_dur),
        "var_95_pct": round(q05, 4),
        "cvar_95_pct": round(cvar, 4),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "win_rate_pct": round(win, 4),
        "total_commissions": round(total_commissions, 4),
        "total_slippage": round(total_slippage, 4),
        "num_trades": num_trades,
        "initial_value": round(portfolio_values.iloc[0], 2),
        "final_value": round(portfolio_values.iloc[-1], 2),
    }
