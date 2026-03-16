"""Strategy library -- pure functions returning WeightSchedule dicts.

Every strategy: (close, rebal_dates, **params) -> WeightSchedule.
Strategies know nothing about engines; they just compute target weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.engine import WeightSchedule


# ═══════════════════════════════════════════════════════════════════════════
# category A -- representative strategy families
# ═══════════════════════════════════════════════════════════════════════════

def sma_momentum(
    close: pd.DataFrame,
    rebal_dates: set[pd.Timestamp],
    *,
    lookback: int = 200,
) -> WeightSchedule:
    """price > SMA(lookback) -> equal weight among passing assets; all below -> cash."""
    sma = close.rolling(lookback, min_periods=lookback).mean()
    tickers = close.columns.tolist()
    ws: WeightSchedule = {}
    for d in sorted(rebal_dates):
        if d not in close.index:
            continue
        if sma.loc[d].isna().any():
            continue
        above = [t for t in tickers if close.loc[d, t] > sma.loc[d, t]]
        if above:
            w = 1.0 / len(above)
            ws[d] = {t: (w if t in above else 0.0) for t in tickers}
        else:
            ws[d] = {t: 0.0 for t in tickers}
    return ws


def inverse_volatility(
    close: pd.DataFrame,
    rebal_dates: set[pd.Timestamp],
    *,
    vol_window: int = 60,
) -> WeightSchedule:
    """w_i = (1/sigma_i) / sum(1/sigma_j); sigma = std of daily returns over vol_window days."""
    rets = close.pct_change()
    vol = rets.rolling(vol_window, min_periods=vol_window).std()
    tickers = close.columns.tolist()
    ws: WeightSchedule = {}
    for d in sorted(rebal_dates):
        if d not in vol.index or vol.loc[d].isna().any():
            continue
        sigmas = vol.loc[d]
        if (sigmas <= 0).any():
            continue
        inv = 1.0 / sigmas
        total = inv.sum()
        ws[d] = {t: float(inv[t] / total) for t in tickers}
    return ws


def cross_sectional_momentum(
    close: pd.DataFrame,
    rebal_dates: set[pd.Timestamp],
    *,
    formation: int = 252,
    skip: int = 21,
    top_k: int = 2,
) -> WeightSchedule:
    """rank by trailing return (formation days, skip most recent skip days), equal-weight top_k."""
    tickers = close.columns.tolist()
    ws: WeightSchedule = {}
    for d in sorted(rebal_dates):
        if d not in close.index:
            continue
        loc = close.index.get_loc(d)
        start = loc - formation - skip
        end = loc - skip
        if start < 0 or end < 0:
            continue
        past = close.iloc[start:end + 1]
        mom = (past.iloc[-1] / past.iloc[0]) - 1.0
        ranked = mom.sort_values(ascending=False)
        winners = ranked.index[:top_k].tolist()
        w = 1.0 / top_k
        ws[d] = {t: (w if t in winners else 0.0) for t in tickers}
    return ws


def ml_signal(
    close: pd.DataFrame,
    rebal_dates: set[pd.Timestamp],
    *,
    model_factory=None,
    train_months: int = 6,
    top_k: int = 2,
) -> WeightSchedule:
    """walk-forward ML: predict 21-day forward return, equal-weight top_k assets.

    model_factory is a zero-arg callable returning a fresh sklearn regressor
    (must have .fit / .predict).  defaults to GBR(50 trees, depth 3).
    features: ret_21, ret_63, ret_126 (cumulative return), vol_20, vol_60.
    training window = train_months * 21 days; 21-day gap prevents lookahead.
    """
    if model_factory is None:
        from sklearn.ensemble import GradientBoostingRegressor
        model_factory = lambda: GradientBoostingRegressor(
            n_estimators=50, max_depth=3, random_state=42,
        )

    tickers = close.columns.tolist()
    rets = close.pct_change()
    warmup = 126

    feats = {}
    for t in tickers:
        df = pd.DataFrame(index=close.index)
        df["ret_21"] = rets[t].rolling(21).sum()
        df["ret_63"] = rets[t].rolling(63).sum()
        df["ret_126"] = rets[t].rolling(126).sum()
        df["vol_20"] = rets[t].rolling(20).std()
        df["vol_60"] = rets[t].rolling(60).std()
        feats[t] = df

    fwd_ret = {t: rets[t].rolling(21).sum().shift(-21) for t in tickers}
    sorted_dates = sorted(d for d in rebal_dates if d in close.index)

    ws: WeightSchedule = {}
    for d in sorted_dates:
        loc = close.index.get_loc(d)
        train_end = loc - 21
        train_start = loc - train_months * 21
        if train_start < warmup or train_end <= warmup:
            continue

        preds = {}
        for t in tickers:
            X_all = feats[t].iloc[train_start:train_end]
            y_all = fwd_ret[t].iloc[train_start:train_end]
            mask = X_all.notna().all(axis=1) & y_all.notna()
            X_train, y_train = X_all.loc[mask].values, y_all.loc[mask].values
            if len(X_train) < 30:
                continue
            model = model_factory()
            model.fit(X_train, y_train)
            x_now = feats[t].loc[[d]]
            if x_now.isna().any(axis=1).iloc[0]:
                continue
            preds[t] = float(model.predict(x_now.values)[0])

        if len(preds) < top_k:
            continue
        ranked = sorted(preds, key=preds.get, reverse=True)
        winners = ranked[:top_k]
        w = 1.0 / top_k
        ws[d] = {t: (w if t in winners else 0.0) for t in tickers}

    return ws


# ═══════════════════════════════════════════════════════════════════════════
# category B -- ablation / stress-test scenarios
# ═══════════════════════════════════════════════════════════════════════════

def daily_binary_switch(
    close: pd.DataFrame,
    rebal_dates: set[pd.Timestamp],
    *,
    asset_a: str | None = None,
    asset_b: str | None = None,
) -> WeightSchedule:
    """alternate 100% in asset_a and 100% in asset_b on each rebalance day."""
    tickers = close.columns.tolist()
    a = asset_a or tickers[0]
    b = asset_b or tickers[1]
    ws: WeightSchedule = {}
    toggle = True
    for d in sorted(rebal_dates):
        if d not in close.index:
            continue
        if toggle:
            ws[d] = {t: (1.0 if t == a else 0.0) for t in tickers}
        else:
            ws[d] = {t: (1.0 if t == b else 0.0) for t in tickers}
        toggle = not toggle
    return ws


def cash_starved_settlement(
    close: pd.DataFrame,
    rebal_dates: set[pd.Timestamp],
    *,
    assets: list[str] | None = None,
) -> WeightSchedule:
    """rotate [60/30/10] <-> [10/30/60] across 3 assets; stresses intra-bar cash settlement."""
    tickers = close.columns.tolist()
    picked = assets or tickers[:3]
    a, b, c = picked[0], picked[1], picked[2]
    rest = [t for t in tickers if t not in picked]
    ws: WeightSchedule = {}
    toggle = True
    for d in sorted(rebal_dates):
        if d not in close.index:
            continue
        w = {a: 0.60, b: 0.30, c: 0.10} if toggle else {a: 0.10, b: 0.30, c: 0.60}
        for t in rest:
            w[t] = 0.0
        ws[d] = w
        toggle = not toggle
    return ws


def concentrated_cascade(
    close: pd.DataFrame,
    rebal_dates: set[pd.Timestamp],
) -> WeightSchedule:
    """95% in one asset, 1.25% in each other; alternates which asset is concentrated."""
    tickers = close.columns.tolist()
    tail_w = (1.0 - 0.95) / (len(tickers) - 1)
    ws: WeightSchedule = {}
    toggle = True
    for d in sorted(rebal_dates):
        if d not in close.index:
            continue
        if toggle:
            ws[d] = {t: (0.95 if t == tickers[0] else tail_w) for t in tickers}
        else:
            ws[d] = {t: (0.95 if t == tickers[-1] else tail_w) for t in tickers}
        toggle = not toggle
    return ws


# ═══════════════════════════════════════════════════════════════════════════
# category C -- frequency amplification
# ═══════════════════════════════════════════════════════════════════════════

def daily_equal_weight(
    close: pd.DataFrame,
    rebal_dates: set[pd.Timestamp],
) -> WeightSchedule:
    """equal-weight (1/N) every trading day; tests divergence amplification from frequency."""
    tickers = close.columns.tolist()
    w = 1.0 / len(tickers)
    ws: WeightSchedule = {}
    for d in sorted(rebal_dates):
        if d not in close.index:
            continue
        ws[d] = {t: w for t in tickers}
    return ws
