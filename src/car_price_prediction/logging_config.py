"""Logging setup shared by CLI commands and the FastAPI application."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from car_price_prediction import config


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
CONSOLE_HANDLER_NAME = "car_price_console"
FILE_HANDLER_NAME = "car_price_file"


def setup_logging(log_level: int = logging.INFO) -> None:
    """Configure console and rotating file logging once per Python process."""

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if not has_handler(root_logger, logging.StreamHandler, CONSOLE_HANDLER_NAME):
        console_handler = logging.StreamHandler()
        console_handler.set_name(CONSOLE_HANDLER_NAME)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if not has_handler(root_logger, RotatingFileHandler, FILE_HANDLER_NAME):
        file_handler = RotatingFileHandler(
            config.LOG_FILE_PATH,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.set_name(FILE_HANDLER_NAME)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def has_handler(
    logger: logging.Logger,
    handler_type: type[logging.Handler],
    handler_name: str,
) -> bool:
    """Return whether a project-owned handler is already registered."""

    return any(
        isinstance(handler, handler_type) and handler.name == handler_name
        for handler in logger.handlers
    )
