import os
import logging
import logging.config

from super_gradients.common.auto_logging import AutoLoggerConfig


# Controlling the default logging level via environment variable
DEFAULT_LOGGING_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Set the default level for all libraries - including 3rd party packages
logging.basicConfig(level=DEFAULT_LOGGING_LEVEL)


def get_logger(
    logger_name: str, training_log_path=None, logs_dir_path=None, log_level=DEFAULT_LOGGING_LEVEL
) -> logging.Logger:
    config_dict = AutoLoggerConfig.generate_config_for_module_name(
        module_name=logger_name, training_log_path=training_log_path, logs_dir_path=logs_dir_path, log_level=log_level
    )
    logging.config.dictConfig(config_dict)
    logger: logging.Logger = logging.getLogger(logger_name)

    if int(os.getenv("LOCAL_RANK", -1)) >= 1:
        shutdown_all_logs()

    return logger


class ILogger:
    """
    Provides logging capabilities to the derived class.
    """

    def __init__(self, logger_name: str = None):
        logger_name = logger_name if logger_name else str(self.__module__)
        self._logger: logging.Logger = get_logger(logger_name)


def shutdown_all_logs():
    # Ignore warnings
    import warnings
    warnings.filterwarnings("ignore")

    # Ignore prints
    import sys
    sys.stdout = open(os.devnull, 'w')  # silent all printing for non master process

    # Only show errors
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.ERROR)
