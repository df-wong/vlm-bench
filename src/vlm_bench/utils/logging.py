"""Structured logging with rich formatting."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
) -> logging.Logger:
    """Configure structured logging."""
    logger = logging.getLogger("vlm_bench")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    # Rich console handler
    rich_handler = RichHandler(
        console=Console(stderr=True),
        show_time=True,
        show_path=False,
        markup=True,
    )
    rich_handler.setLevel(getattr(logging, level.upper()))
    logger.addHandler(rich_handler)

    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
