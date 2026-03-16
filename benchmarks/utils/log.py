"""unified logging setup for the benchmark suite."""

import logging
import sys

_CONFIGURED = False

_NOISY_LIBS = (
    "cvxportfolio",
    "numexpr",
    "matplotlib",
    "PIL",
    "urllib3",
    "fsspec",
    "pyarrow",
    "h5py",
)


def setup_logging(level: int = logging.INFO) -> None:
    """configure root logger with a clean format; safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(name)-30s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    for lib in _NOISY_LIBS:
        logging.getLogger(lib).setLevel(logging.WARNING)
