# crossengine

multi-asset portfolio backtesting engine + cross-engine concordance testing.

companion code for the paper "Implementation Risk in Portfolio Backtesting: A Previously Unquantified Source of Error" (Yin, Miki, Lesnichenko).

## install

```bash
pip install -e .
```

## input data

both the engine and concordance testing take the same input: a pandas DataFrame of daily closing prices. rows are trading dates, columns are ticker symbols.

```
              AAPL    MSFT
2020-01-02  300.35  158.78
2020-01-03  297.43  159.03
2020-01-06  299.80  159.95
...
2024-12-31  254.49  421.50
```

you can load this from a CSV, a parquet, or build it with yfinance:

```python
import yfinance as yf
prices = yf.download(["AAPL", "MSFT"], start="2020-01-01", end="2025-01-01")["Close"]
```

## 1. backtesting engine

the engine takes two DataFrames: `prices` (daily closes) and `signals` (target weights on rebalance dates).

```python
import pandas as pd
from crossengine import backtest

# prices: rows = dates, columns = tickers, values = close prices
prices = pd.DataFrame({
    "AAPL": [300, 297, 299, 305, 310],
    "MSFT": [158, 159, 160, 162, 165],
}, index=pd.bdate_range("2020-01-02", periods=5))

# signals: rows = rebalance dates, columns = tickers, values = target weights
# only include rows for dates you want to rebalance; other dates hold automatically
signals = pd.DataFrame({
    "AAPL": [0.6],
    "MSFT": [0.4],
}, index=[prices.index[0]])  # rebalance once on day 0, then hold

result = backtest(prices, signals, initial_cash=10_000, commission=0.001)
print(result.report)
result.plot()
```

signal values are target weights. they are auto-normalised to sum to 1. rows in signals trigger a rebalance on that date; dates without a row hold the existing positions.

### hold without rebalancing (STAY)

STAY freezes the share count, not the weight. price movement causes the weight to drift naturally. no trades are generated for STAY assets.

```python
from crossengine import backtest, STAY
import pandas as pd

dates = pd.bdate_range("2020-01-02", periods=3)
prices = pd.DataFrame({
    "AAPL": [300, 310, 320],
    "MSFT": [158, 155, 160],
}, index=dates)

signals = pd.DataFrame({
    "AAPL": [0.6, STAY, 0.3],
    "MSFT": [0.4, STAY, 0.7],
}, index=dates)

# day 0: allocate 60% AAPL, 40% MSFT
# day 1: STAY -- share counts frozen, weights drift with price
# day 2: rebalance to 30% AAPL, 70% MSFT

result = backtest(prices, signals, initial_cash=10_000, commission=0.001)
```

### partial rebalance

rebalance some assets while letting others drift:

```python
signals = pd.DataFrame({
    "AAPL": [0.6, STAY],   # hold AAPL, let it drift
    "MSFT": [0.4, 0.7],    # rebalance MSFT to 70% of remaining budget
}, index=[dates[0], dates[2]])
```

MSFT is rebalanced from the budget remaining after AAPL's drifted value. this is a common pattern for sector rotation (hold winners, rebalance losers).

## 2. concordance testing

run the same strategy through multiple backtesting engines and measure how much they disagree. the concordance API uses a dict format (not a DataFrame) because it needs to pass the same schedule to engines with different internal formats.

### with a strategy function

```python
import pandas as pd
from crossengine.concordance import concordance

# same prices DataFrame as above
prices = pd.DataFrame({
    "AAPL": [300, 297, 299, 305, 310, 315, 320, 318, 325, 330],
    "MSFT": [158, 159, 160, 162, 165, 163, 168, 170, 172, 175],
}, index=pd.bdate_range("2020-01-02", periods=10))

def equal_weight(close, rebal_dates):
    """allocate equally across all assets on each rebalance date."""
    tickers = close.columns.tolist()
    w = 1.0 / len(tickers)
    result = {}
    for d in rebal_dates:
        result[d] = {t: w for t in tickers}
    return result

report = concordance(equal_weight, prices)
print(report.summary())
```

the strategy function receives `(close, rebal_dates)` and returns a dict:

```python
{
    Timestamp("2020-01-02"): {"AAPL": 0.5, "MSFT": 0.5},
    Timestamp("2020-02-03"): {"AAPL": 0.5, "MSFT": 0.5},
}
```

rebalance dates default to monthly (first trading day of each month). you can override with `rebal_dates=`.

### with pre-computed weights (no function needed)

```python
from crossengine.concordance import concordance

weights = {
    pd.Timestamp("2020-01-02"): {"AAPL": 0.5, "MSFT": 0.5},
    pd.Timestamp("2020-01-08"): {"AAPL": 0.3, "MSFT": 0.7},
}
report = concordance(weights, prices)
```

### with STAY (partial rebalancing)

```python
from crossengine.concordance import concordance, STAY

schedule = {
    pd.Timestamp("2020-01-02"): {"AAPL": 0.6, "MSFT": 0.4},
    pd.Timestamp("2020-01-08"): {"AAPL": STAY, "MSFT": 0.7},
}
report = concordance(schedule, prices)
```

engines that can query live portfolio state (ours, bt, backtrader) resolve STAY natively. engines that receive all signals upfront (vectorbt, cvxportfolio) get pre-resolved drifted weights via forward simulation.

### what you get back

```python
report.summary()          # human-readable text report
report.max_divergence     # max relative divergence across all pairs (%)
report.engine_sensitivity # coefficient of variation of final values (%)
report.equity             # DataFrame of equity curves (columns = engine names)
report.divergence         # dict of pairwise divergence metrics
report.to_json("out.json")
report.plot()
```

example output:

```
engine concordance report
==================================================
engines: ours, bt, vectorbt (3 active)

pairwise divergence (max relative %):
  bt vs ours                       0.1200%
  ours vs vectorbt                 0.0300%
  bt vs vectorbt                   0.0900%

max divergence:      0.1200%
engine sensitivity:  0.0800%

final portfolio values:
  ours                 $110,234.56
  bt                   $110,102.33
  vectorbt             $110,201.78
```

### graceful engine detection

concordance runs whichever engines are installed. if vectorbt is missing, it is skipped:

```python
report = concordance(weights, prices, engines=("ours", "bt"))
```

supported engines: `ours`, `bt`, `vectorbt`, `backtrader`, `cvxportfolio`.

## 3. reproduce paper results

data files are gitignored and must be generated first:

```bash
python data/curation/download_yfinance.py             # download 180 stocks from yfinance
python data/curation/stratification.py                # generate 30 stratified buckets
```

then run the benchmarks:

```bash
cd benchmarks && python run_all.py                    # 15 benchmarks x 5 engines x 30 buckets
cd benchmarks && PYTHONPATH=. python summary/run.py   # regenerate figures
```

## repository layout

```
src/crossengine/                backtesting engine
src/crossengine/concordance/    cross-engine concordance API
benchmarks/                  15 benchmark strategies (BM01-BM12 + ML approaches)
  utils/wrappers/            engine wrappers (bt, vectorbt, backtrader, cvxportfolio)
  summary/                   figure generation + statistical analysis
data/                        180 S&P 500 stocks, 30 stratified buckets
tests/                       44 tests (including 30-bucket x 5-engine e2e)
```

## licence

MIT
