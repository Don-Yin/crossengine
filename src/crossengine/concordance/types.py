"""shared type aliases for concordance testing."""
from __future__ import annotations

from typing import Literal

import pandas as pd

STAY: Literal["STAY"] = "STAY"

SignalSchedule = dict[pd.Timestamp, dict[str, float | Literal["STAY"]]]
WeightSchedule = dict[pd.Timestamp, dict[str, float]]
