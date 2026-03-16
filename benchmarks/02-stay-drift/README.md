# 02-stay-drift

difficulty: moderate

allocate 60/40 AAPL/MSFT with trading212 costs, then drift for 60 days (STAY -- shares frozen, weights drift with price), then partial rebalance (STAY AAPL, reallocate MSFT to remaining budget). compared against `bt` framework (commission only -- bt has no slippage model).

what it proves -- STAY freezes share count not weight. bt naturally drifts between rebalances, so a custom bt algo that reads AAPL's drifted weight and only rebalances MSFT replicates our STAY semantics exactly. validates that our engine and bt agree on how drift + partial rebalance interact.

expected result -- <0.5% relative difference (divergence from slippage only).
