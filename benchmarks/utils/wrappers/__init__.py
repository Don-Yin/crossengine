"""engine wrapper adapters: each file implements one backtesting engine adapter.

category A (native STAY): ours, bt, backtrader -- accept SignalSchedule
category B (pre-resolved): vectorbt, cvxportfolio -- accept WeightSchedule

excluded engines (zipline, nautilus) are kept as forensic reference but
not imported here. see their module docstrings for exclusion reasons.
"""

from utils.wrappers.ours import run_ours
from utils.wrappers.bt_engine import run_bt_engine
from utils.wrappers.vectorbt_engine import run_vbt_engine
from utils.wrappers.backtrader_engine import run_backtrader_engine
from utils.wrappers.cvxportfolio_engine import run_cvxportfolio_engine

ACTIVE_ENGINES: tuple[str, ...] = ("ours", "bt", "vectorbt", "backtrader", "cvxportfolio")

__all__ = [
    "ACTIVE_ENGINES",
    "run_ours",
    "run_bt_engine",
    "run_vbt_engine",
    "run_backtrader_engine",
    "run_cvxportfolio_engine",
]
