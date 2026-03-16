# 04-rotation-with-cost

difficulty: hardest

same alternating 80/5/5/5/5 strategy as 03-rotation but with 2x trading212 costs -- 0.30% commission, 6 bps spread. a stress test for cost scaling. compared against `bt` framework (commission only -- bt has no slippage model).

what it proves -- cost models scale correctly under heavy friction. doubles the rates of 03-rotation. the larger costs amplify any order-of-operations bugs and test the cash-scaling logic under tighter budgets.

expected result -- <3% relative difference (divergence from slippage + commission order-of-ops).
