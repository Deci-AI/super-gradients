import logging
import logging.config

from super_gradients.common.auto_logging import AutoLoggerConfig
from super_gradients.common.environment.environment_config import DEFAULT_LOGGING_LEVEL


def get_logger(
    logger_name: str, training_log_path=None, logs_dir_path=None, log_level=DEFAULT_LOGGING_LEVEL
) -> logging.Logger:
    config_dict = AutoLoggerConfig.generate_config_for_module_name(
        module_name=logger_name, training_log_path=training_log_path, logs_dir_path=logs_dir_path, log_level=log_level
    )
    logging.config.dictConfig(config_dict)
    logger: logging.Logger = logging.getLogger(logger_name)
    return logger


class ILogger:
    """
    Provides logging capabilities to the derived class.
    """

    def __init__(self, logger_name: str = None):
        logger_name = logger_name if logger_name else str(self.__module__)
        self._logger: logging.Logger = get_logger(logger_name)
