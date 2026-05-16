from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str = "akkadian_mt") -> logging.Logger:
    """Return a singleton-style logger writing both to console and data/log_file.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    log_dir = Path("data")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log_file.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
