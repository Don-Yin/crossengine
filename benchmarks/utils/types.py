"""shared type aliases for the benchmarking framework."""
from __future__ import annotations

import pandas as pd

WeightSchedule = dict[pd.Timestamp, dict[str, float]]
