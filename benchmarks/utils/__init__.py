"""benchmarks.utils -- shared facilities for all benchmark scripts.

modules
-------
data          constants, data loaders, rebalance-date helpers
engine        SignalSchedule type, engine adapters, unified runner
comparison    write_comparison, publication figure styling
strategies    strategy library (pure functions -> SignalSchedule)
"""

from utils.log import setup_logging

setup_logging()

from utils.comparison import (
    ENGINE_COLORS,
    ENGINE_LINESTYLES,
    ENGINE_LINEWIDTHS,
    pub_style,
    write_comparison,
)
from utils.data import (
    BACKTEST_START,
    DATA_DIR,
    RESULTS_ROOT,
    ROOT,
    T212_COMMISSION,
    T212_SLIPPAGE,
    alternating_weights,
    every_day,
    load_close,
    load_close_full,
    load_spx,
    month_start_dates,
    month_starts,
)
from utils.engine import (
    WeightSchedule,
    run_benchmark,
    run_bt_engine,
    run_ours,
    run_vbt_engine,
)
from utils.runner import load_buckets, run_multi_bucket
