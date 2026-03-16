# 03-rotation

difficulty: hard

alternates monthly between 80/5/5/5/5 and 5/5/5/5/80 across 5 assets with trading212 costs -- 0.15% fx fee, ~3 bps spread. compared against `bt` framework (commission only -- bt has no slippage model).

what it proves -- engine handles large portfolio rotations correctly with realistic friction. on each rebalance ~75% of portfolio value changes hands. this exercises sells-first-then-buys ordering, cash scaling, and slippage compounding over many rotations.

expected result -- <1.5% relative difference (divergence attributable to slippage).
