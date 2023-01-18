import os
import logging
import logging.config
from typing import Union

from super_gradients.common.auto_logging.auto_logger import AutoLoggerConfig


def get_logger(logger_name: str, log_level: Union[str, None] = None) -> logging.Logger:
    AutoLoggerConfig.get_instance()
    logger: logging.Logger = logging.getLogger(logger_name)
    if log_level is not None:
        logger.setLevel(log_level)
    if int(os.getenv("LOCAL_RANK", -1)) > 0:
        mute_current_process()
    return logger


class ILogger:
    """
    Provides logging capabilities to the derived class.
    """

    def __init__(self, logger_name: str = None):
        logger_name = logger_name if logger_name else str(self.__module__)
        self._logger: logging.Logger = get_logger(logger_name)


def mute_current_process():
    """Mute prints, warnings and all logs except ERRORS. This is meant when running multiple processes."""
    # Ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    # Ignore prints
    import sys

    sys.stdout = open(os.devnull, "w")

    # Only show ERRORS
    process_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in process_loggers:
        logger.setLevel(logging.ERROR)
