# API reference

## backtesting engine

### `backtest(prices, signals, **kwargs)`

run a multi-asset portfolio backtest.

```python
from crossengine import backtest

result = backtest(
    prices,                    # DataFrame: rows=dates, columns=tickers, values=close prices
    signals,                   # DataFrame: rows=rebalance dates, columns=tickers, values=target weights or STAY
    initial_cash=10_000,       # starting cash balance
    commission=0.001,          # proportional commission rate (0.001 = 0.1%)
    slippage=0.0,              # proportional slippage rate
    long_only=True,            # clip negative signals to zero
    risk_free=0.04,            # annual risk-free rate for Sharpe/Sortino
)
```

**parameters:**

| parameter | type | default | description |
|-----------|------|---------|-------------|
| `prices` | `DataFrame` or `OHLCV` | required | daily prices. plain DataFrame treated as close-only |
| `signals` | `DataFrame` | required | target weights. rows trigger rebalance; absent dates hold |
| `orders` | `OrderQueue` | `None` | optional limit/stop orders |
| `initial_cash` | `float` | `10_000` | starting cash |
| `commission` | `float` or `CommissionModel` | `0.001` | proportional rate or custom model |
| `slippage` | `float` or `SlippageModel` | `0.0` | proportional rate or custom model |
| `long_only` | `bool` | `True` | clip negative signals to zero |
| `risk_free` | `float` | `0.04` | annual risk-free rate |

**returns:** `BacktestResult`

Current execution semantics are same-bar close fills: when a date is present in
`signals`, the rebalance is applied on that same date using the corresponding
close from `prices`. This is a close-fill model, not an implicit next-bar-open
execution model.

### `STAY`

sentinel value for "freeze share count, let weight drift." use in signals DataFrame:

```python
from crossengine import STAY

signals = pd.DataFrame({
    "AAPL": [0.6, STAY],   # day 0: allocate 60%, day 1: hold shares
    "MSFT": [0.4, 0.7],    # day 0: allocate 40%, day 1: rebalance to 70%
}, index=[date_0, date_1])
```

### `BacktestResult`

returned by `backtest()`. properties:

| property | type | description |
|----------|------|-------------|
| `portfolio_value` | `Series` | daily portfolio value |
| `cash` | `Series` | daily cash balance |
| `returns` | `Series` | daily returns |
| `trades` | `DataFrame` | trade log (date, asset, side, quantity, price, commission, slippage, type, tag) |
| `metrics` | `dict` | all computed metrics |
| `report` | `str` | human-readable text report |

methods:

| method | description |
|--------|-------------|
| `weights()` | DataFrame of daily portfolio weights |
| `positions()` | DataFrame of daily share counts |
| `plot()` | 5-panel figure (6 panels when benchmark is set) |

### commission models

```python
from crossengine import FlatRate, IBKRTiered, IBKRFixed, NoCommission

result = backtest(prices, signals, commission=IBKRTiered())
```

| model | description |
|-------|-------------|
| `FlatRate(rate, min_fee=0.0)` | proportional: `max(abs(qty) * price * rate, min_fee)` |
| `IBKRTiered()` | IBKR tiered pricing schedule |
| `IBKRFixed()` | IBKR fixed pricing schedule |
| `NoCommission()` | zero commission |

### slippage models

```python
from crossengine import FixedSlippage, VolumeImpact, NoSlippage

result = backtest(prices, signals, slippage=FixedSlippage(0.001))
```

| model | description |
|-------|-------------|
| `FixedSlippage(rate)` | proportional: `price * (1 + rate * sign)` |
| `VolumeImpact(fixed_rate, impact_factor)` | combines fixed proportional rate + volume-weighted impact |
| `NoSlippage()` | no slippage |

---

## concordance testing

### `concordance(strategy, close, **kwargs)`

run a strategy through multiple engines and measure concordance.

```python
from crossengine.concordance import concordance

report = concordance(
    strategy,                  # SignalSchedule dict or callable
    close,                     # DataFrame of close prices
    rebal_dates=None,          # set of rebalance dates (default: monthly)
    initial_cash=100_000,      # starting cash
    commission=0.0015,         # proportional commission rate
    slippage=0.0003,           # proportional slippage rate
    engines=None,              # tuple of engine names (default: all installed)
)
```

**parameters:**

| parameter | type | default | description |
|-----------|------|---------|-------------|
| `strategy` | `dict` or `callable` | required | `SignalSchedule` dict or `(close, rebal_dates) -> SignalSchedule` |
| `close` | `DataFrame` | required | daily close prices |
| `rebal_dates` | `set[Timestamp]` | `None` | rebalance dates (default: first trading day of each month) |
| `initial_cash` | `float` | `100_000` | starting cash |
| `commission` | `float` | `0.0015` | proportional commission rate |
| `slippage` | `float` | `0.0003` | proportional slippage rate |
| `engines` | `tuple[str, ...]` | `None` | engines to use (default: all installed) |

**returns:** `ConcordanceReport`

### `SignalSchedule`

the interchange format for concordance testing:

```python
SignalSchedule = dict[pd.Timestamp, dict[str, float | Literal["STAY"]]]
```

examples:

```python
# pure weights (no STAY)
schedule = {
    pd.Timestamp("2020-01-02"): {"AAPL": 0.5, "MSFT": 0.5},
    pd.Timestamp("2020-02-03"): {"AAPL": 0.3, "MSFT": 0.7},
}

# with STAY (partial rebalancing)
from crossengine.concordance import STAY

schedule = {
    pd.Timestamp("2020-01-02"): {"AAPL": 0.6, "MSFT": 0.4},
    pd.Timestamp("2020-04-01"): {"AAPL": STAY, "MSFT": 0.7},
}
```

dates absent from the dict = hold everything. STAY in a value = hold that asset while rebalancing others.

### `ConcordanceReport`

returned by `concordance()`. properties:

| property | type | description |
|----------|------|-------------|
| `engines` | `list[str]` | engines that ran |
| `equity` | `DataFrame` | equity curves (columns = engine names) |
| `divergence` | `dict` | pairwise divergence metrics |
| `max_divergence` | `float` | max relative divergence across all pairs (%) |
| `engine_sensitivity` | `float` | coefficient of variation of final values (%) |

methods:

| method | description |
|--------|-------------|
| `summary()` | human-readable text report |
| `to_json(path)` | write report to JSON |
| `plot()` | equity curves + divergence plot |

### supported engines

| engine | package | category | STAY support |
|--------|---------|----------|-------------|
| `ours` | `crossengine` (this package) | A (native) | native |
| `bt` | `bt` | A (native) | native via `target.children` |
| `backtrader` | `backtrader` | A (native) | native via `getposition()` |
| `vectorbt` | `vectorbt` | B (pre-resolved) | via `resolve_stay()` |
| `cvxportfolio` | `cvxportfolio` | B (pre-resolved) | via `resolve_stay()` |

category A engines resolve STAY at runtime using live portfolio state. category B engines receive pre-computed drifted weights from a forward simulation.
