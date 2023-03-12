import logging
import logging.config
from typing import Union

from super_gradients.common.abstractions.mute_processes import mute_subprocesses
from super_gradients.common.auto_logging.auto_logger import AutoLoggerConfig

# Mute on import to avoid the import prints/logs on sub processes
mute_subprocesses()


def get_logger(logger_name: str, log_level: Union[str, None] = None) -> logging.Logger:
    AutoLoggerConfig.get_instance()
    logger: logging.Logger = logging.getLogger(logger_name)
    if log_level is not None:
        logger.setLevel(log_level)

    mute_subprocesses()
    return logger


class ILogger:
    """
    Provides logging capabilities to the derived class.
    """

    def __init__(self, logger_name: str = None):
        logger_name = logger_name if logger_name else str(self.__module__)
        self._logger: logging.Logger = get_logger(logger_name)
