"""
Setup logging utilities.
"""

import logging
from logging import Logger
import os


class LoggingConstants:
    """Logging constant values"""

    LOGGER_NAME = "ranker"
    FNAME = "ranker.log"
    LOG_FORMAT = "%(levelname)s: %(asctime)s.%(msecs)03d \n%(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    name: str = LoggingConstants.LOGGER_NAME,
    reset: bool = False,
    file_name: str = LoggingConstants.FNAME,
    log_to_file: bool = True,
) -> Logger:

    if not name:
        name = os.path.basename(__file__).split(".")[0]
    logger = logging.getLogger(name)

    if reset:
        map(logger.removeHandler, logger.handlers[:])
        map(logger.removeFilter, logger.filters[:])

    if len(logger.handlers) >= 1:
        return logger

    if log_to_file:
        file_handler = logging.FileHandler(file_name, mode="w")
        file_handler.setFormatter(
            logging.Formatter(
                fmt=LoggingConstants.LOG_FORMAT, datefmt=LoggingConstants.LOG_DATE_FORMAT
            )
        )
        logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    try:
        from colorlog import ColoredFormatter

        color_formatter = ColoredFormatter(
            "%(log_color)s" + LoggingConstants.LOG_FORMAT, datefmt=LoggingConstants.LOG_DATE_FORMAT
        )
    except ImportError:
        color_formatter = logging.Formatter(
            fmt=LoggingConstants.LOG_FORMAT, datefmt=LoggingConstants.LOG_DATE_FORMAT
        )
    stream_handler.setFormatter(color_formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)

    return logger


def set_log_level(debug, logger) -> Logger:
    """
    Sets StreamHandler to debug level.
    """
    if debug > 0:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)
    if debug > 1:
        logger.setLevel(1)
    return logger
