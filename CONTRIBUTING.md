# contributing

contributions are welcome. this project has two components:

1. **backtesting engine** (`src/crossengine/`) -- the core engine
2. **concordance testing** (`src/crossengine/concordance/`) -- cross-engine validation

## setup

```bash
git clone https://github.com/donyin/crossengine.git
cd crossengine
pip install -e ".[dev]"
```

## running tests

```bash
python -m pytest tests/ -v
```

the full e2e test (30 buckets x 5 engines) requires all engine packages installed:

```bash
pip install bt vectorbt backtrader cvxportfolio
python -m pytest tests/test_e2e_full.py -v
```

## adding a new engine wrapper

1. add a `run_<engine>_engine()` function to `src/crossengine/concordance/engines.py`
2. the function must accept `(close: DataFrame, ss: SignalSchedule, *, initial_cash, commission)` for category A engines (native STAY) or `(close: DataFrame, ws: WeightSchedule, *, initial_cash, commission)` for category B engines (pre-resolved)
3. return a `pd.Series` of daily portfolio values
4. add the engine to `detect_engines()` and the dispatch logic in `api.py`
5. add tests

## code style

- python 3.11+
- no try/except blocks -- expose errors, fix the root cause
- files under 250 LOC
- prefer `pathlib.Path` over `os.path`
- single-line docstrings, lowercase initials
- no monkey patches

## reporting issues

open an issue on GitHub with:
- what you expected
- what happened
- minimal reproduction steps
