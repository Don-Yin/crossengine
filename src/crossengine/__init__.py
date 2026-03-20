"""crossengine -- multi-asset portfolio backtesting engine with cross-engine concordance testing.

Quick start::

    import pandas as pd
    from crossengine import backtest, STAY

    prices = pd.DataFrame({"AAPL": [...], "MSFT": [...]}, index=dates)
    signals = pd.DataFrame({"AAPL": [0.6, "s", 0.3], "MSFT": [0.4, "s", 0.7]}, index=dates)

    result = backtest(prices, signals, commission=0.001, slippage=0.001)
    print(result.metrics)
    result.plot()
"""

from .data import OHLCV
from .engine import backtest
from .models.commission import (
    CommissionModel,
    FlatRate,
    IBKRFixed,
    IBKRTiered,
    NoCommission,
)
from .models.slippage import (
    FixedSlippage,
    NoSlippage,
    SlippageModel,
    VolumeImpact,
)
from .orders import Order, OrderQueue, OrderType
from .portfolio import Portfolio
from .result import BacktestResult
from .signals import STAY

__all__ = [
    "backtest",
    "OHLCV",
    "STAY",
    "BacktestResult",
    "Portfolio",
    "Order",
    "OrderQueue",
    "OrderType",
    "CommissionModel",
    "FlatRate",
    "IBKRTiered",
    "IBKRFixed",
    "NoCommission",
    "SlippageModel",
    "FixedSlippage",
    "VolumeImpact",
    "NoSlippage",
]
