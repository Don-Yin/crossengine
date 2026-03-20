"""Signal resolution with STAY handling.

STAY semantics: freezes the **share count**, not the weight.  Price movement
causes the weight to drift naturally -- that is intentional and no trades are
generated for STAY assets.  Active (non-STAY) signals are normalised into the
remaining portfolio budget, which is ``total_value`` minus the *current market
value* of all STAY positions.
"""

from __future__ import annotations

STAY = "s"


def resolve_signals(
    raw: dict[str, float | str],
    current_shares: dict[str, float],
    prices: dict[str, float],
    total_value: float,
    long_only: bool = True,
) -> dict[str, float]:
    """Resolve one bar's signal row into *target share counts*.

    Parameters
    ----------
    raw
        Mapping of asset → signal value.  A numeric value is an allocation
        signal; the sentinel ``STAY`` (``"s"``) means "keep the current
        number of shares -- do not trade this asset".
    current_shares
        Current share holdings per asset.
    prices
        Current bar prices per asset.
    total_value
        Portfolio total value (cash + equity) *before* this bar's trades.
    long_only
        When ``True`` negative signals are clipped to 0.

    Returns
    -------
    dict[str, float]
        Target share count per asset.  The caller computes deltas against
        *current_shares* to determine what trades to execute.
    """
    target: dict[str, float] = {}
    stay_value = 0.0

    # ── Phase 1: lock STAY assets at their current share count ──────────
    for asset, sig in raw.items():
        if sig == STAY:
            held = current_shares.get(asset, 0.0)
            target[asset] = held
            stay_value += held * prices.get(asset, 0.0)

    # ── Phase 2: budget available for active signals ────────────────────
    budget = total_value - stay_value

    active: dict[str, float] = {}
    for asset, sig in raw.items():
        if sig != STAY:
            val = float(sig)
            active[asset] = max(val, 0.0) if long_only else val

    if not active:
        return target

    # STAY positions might exceed total value after a large price rally in
    # stay assets combined with losses elsewhere -- nothing left to allocate.
    if budget <= 0:
        for asset in active:
            target[asset] = current_shares.get(asset, 0.0)
        return target

    # ── Phase 3: normalise active signals into the remaining budget ─────
    signal_sum = sum(abs(v) for v in active.values())

    if signal_sum > 0:
        for asset, sig_val in active.items():
            proportion = abs(sig_val) / signal_sum
            target_value = proportion * budget
            p = prices.get(asset, 0.0)
            target[asset] = target_value / p if p > 0 else 0.0
    else:
        for asset in active:
            target[asset] = 0.0

    return target
