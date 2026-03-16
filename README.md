# implementation risk in portfolio backtesting

companion code for the paper "Implementation Risk in Portfolio Backtesting: A Previously Unquantified Source of Error" (Yin, Miki, Lesnichenko).

## what this repository contains

- a purpose-built reference backtesting engine (`src/backtest/`) whose cost model is a direct translation of the proportional-cost specification in Algorithm 1 of the paper
- 15 benchmark strategies across five categories (simple, signal, ML, rotation, ablation) defined in `benchmarks/`
- engine wrappers for bt, vectorbt, Backtrader, and cvxportfolio that align each library to the same data feed, rebalancing calendar, and cost specification
- stratified asset-bucket construction via Mahalanobis rerandomisation over 180 S&P 500 constituents (`data/curation/`)
- figure generation and statistical analysis scripts (`benchmarks/summary/`)

## quick start

```bash
micromamba activate algotrade
pip install -e .
```

run all 15 benchmarks through all five engines:

```bash
cd benchmarks && python run_all.py
```

regenerate figures:

```bash
cd benchmarks && PYTHONPATH=. python summary/run.py
```

## repository layout

```
src/backtest/          reference engine (Algorithm 1)
benchmarks/
  strategies/          15 benchmark definitions (BM01-BM12)
  utils/wrappers/      engine wrappers (bt, vectorbt, backtrader, cvxportfolio)
  summary/figures/     plotting modules (Nature-style colour scheme)
data/curation/         universe selection, bucket construction, QC
```

## paper

the manuscript source lives in a separate Overleaf-synced directory. the LaTeX files use the Springer Nature `sn-jnl` template.

## licence

MIT
