"""
Setup logging utilities.
"""
import logging


class LoggingConstants:
    """Logging constant values"""

    LOGGER_NAME = "ml4ir"
    FILE_NAME = "ml4ir.log"
    LOG_FORMAT = "[%(levelname)s] %(asctime)s.%(msecs)03d: %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(file_name: str = LoggingConstants.FILE_NAME) -> logging.Logger:
    """
    Setup logging handlers to file and stdout

    Parameters
    ----------
    file_name: str
        Output file to write logs to

    Returns
    -------
    logger
        Logging handler
    """
    logger = logging.getLogger(LoggingConstants.LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt=LoggingConstants.LOG_FORMAT,
        datefmt=LoggingConstants.LOG_DATE_FORMAT
    )

    # Log to file
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log to stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
