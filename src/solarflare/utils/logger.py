import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Create and return a configured application logger."""

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logger.setLevel(log_level)

    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger