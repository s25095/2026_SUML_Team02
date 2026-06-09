from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from car_price_prediction import config


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_level: int = logging.INFO) -> None:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if not has_handler(root_logger, logging.StreamHandler, marker="console"):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        console_handler._car_price_handler = "console"
        root_logger.addHandler(console_handler)

    if not has_handler(root_logger, RotatingFileHandler, marker="file"):
        file_handler = RotatingFileHandler(
            config.LOG_FILE_PATH,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler._car_price_handler = "file"
        root_logger.addHandler(file_handler)


def has_handler(
    logger: logging.Logger,
    handler_type: type[logging.Handler],
    marker: str,
) -> bool:
    return any(
        isinstance(handler, handler_type)
        and getattr(handler, "_car_price_handler", None) == marker
        for handler in logger.handlers
    )
