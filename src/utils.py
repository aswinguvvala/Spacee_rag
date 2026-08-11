"""Shared helpers used across the pipeline: logging config and project paths.

Every other module in ``src`` imports ``get_logger`` from here instead of
calling ``print`` or configuring ``logging`` itself, so log format/level stays
consistent everywhere and can be changed in one place.
"""
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CORPUS_DIR: Path = PROJECT_ROOT / "corpus"
INDEX_DIR: Path = PROJECT_ROOT / "corpus" / "index"
EVAL_DIR: Path = PROJECT_ROOT / "eval"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with consistent formatting.

    Configures the root logger exactly once (idempotent across repeated
    calls/imports), then returns a child logger for ``name``.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stderr,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
