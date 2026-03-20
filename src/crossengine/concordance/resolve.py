"""forward-simulate a portfolio to materialize STAY into concrete drifted weights."""
from __future__ import annotations

import pandas as pd

from crossengine.concordance.types import STAY, SignalSchedule, WeightSchedule


def resolve_stay(
    ss: SignalSchedule,
    close: pd.DataFrame,
    initial_cash: float,
    cost_rate: float = 0.0,
) -> WeightSchedule:
    """resolve STAY signals into concrete drifted weights by forward simulation."""
    tickers = close.columns.tolist()
    signal_dates = sorted(ss.keys())
    cash = initial_cash
    shares: dict[str, float] = {t: 0.0 for t in tickers}
    ws: WeightSchedule = {}

    for d in signal_dates:
        if d not in close.index:
            continue

        prices = {t: float(close.loc[d, t]) for t in tickers}
        total_value = cash + sum(shares[t] * prices[t] for t in tickers)

        if total_value <= 0:
            ws[d] = {t: 0.0 for t in tickers}
            continue

        raw = ss[d]
        stay_assets: list[str] = []
        active: dict[str, float] = {}
        for t in tickers:
            sig = raw.get(t, 0.0)
            if sig == STAY:
                stay_assets.append(t)
            else:
                active[t] = float(sig)

        stay_value = sum(shares[t] * prices[t] for t in stay_assets)
        budget = total_value - stay_value
        active_sum = sum(abs(v) for v in active.values())
        new_shares: dict[str, float] = {}

        for t in stay_assets:
            new_shares[t] = shares[t]

        if active_sum > 0 and budget > 0:
            for t, sig_val in active.items():
                proportion = abs(sig_val) / active_sum
                target_value = proportion * budget
                new_shares[t] = target_value / prices[t] if prices[t] > 0 else 0.0
        else:
            for t in active:
                new_shares[t] = 0.0

        total_after = sum(new_shares[t] * prices[t] for t in tickers)
        if total_after > 0:
            ws[d] = {t: (new_shares[t] * prices[t]) / total_after for t in tickers}
        else:
            ws[d] = {t: 0.0 for t in tickers}

        for t in tickers:
            old_val = shares[t] * prices[t]
            new_val = new_shares[t] * prices[t]
            trade_val = abs(new_val - old_val)
            cash -= (new_val - old_val) + trade_val * cost_rate

        shares = new_shares

    return ws


def has_stay(ss: SignalSchedule) -> bool:
    """check if a signal schedule contains any STAY sentinels."""
    return any(v == STAY for row in ss.values() for v in row.values())
