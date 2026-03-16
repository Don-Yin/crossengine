"""08 -- run all ML signal approaches across all buckets."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

APPROACHES_DIR = Path(__file__).resolve().parent / "approaches"


def main():
    for script in sorted(APPROACHES_DIR.glob("*.py")):
        name = script.stem
        logger.info("=" * 60)
        logger.info("  running approach: %s", name)
        logger.info("=" * 60)

        spec = importlib.util.spec_from_file_location(name, script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()


if __name__ == "__main__":
    main()
