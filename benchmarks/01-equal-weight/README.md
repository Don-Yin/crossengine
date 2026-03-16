# 01-equal-weight

difficulty: trivial

equal-weight monthly rebalance across 5 assets with trading212 costs -- 0.15% fx fee as commission, ~3 bps bid-ask spread as slippage, fractional shares. compared against `bt` framework (commission only -- bt has no slippage model).

what it proves -- basic weight normalisation, target-share computation, and cost models are correct. small divergence from bt is attributable entirely to our slippage.

expected result -- <0.5% relative difference.
