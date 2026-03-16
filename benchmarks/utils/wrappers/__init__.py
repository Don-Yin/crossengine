"""engine wrapper adapters: each file implements one backtesting engine adapter
that consumes a WeightSchedule and returns a portfolio value pd.Series."""

from utils.wrappers.ours import run_ours
from utils.wrappers.bt_engine import run_bt_engine
from utils.wrappers.vectorbt_engine import run_vbt_engine
from utils.wrappers.backtrader_engine import run_backtrader_engine
from utils.wrappers.cvxportfolio_engine import run_cvxportfolio_engine
from utils.wrappers.zipline_engine import run_zipline_engine
from utils.wrappers.nautilus_engine import run_nautilus_engine

ACTIVE_ENGINES: tuple[str, ...] = ("ours", "bt", "vectorbt", "backtrader", "cvxportfolio")

__all__ = [
    "ACTIVE_ENGINES",
    "run_ours",
    "run_bt_engine",
    "run_vbt_engine",
    "run_backtrader_engine",
    "run_cvxportfolio_engine",
    "run_zipline_engine",
    "run_nautilus_engine",
]
