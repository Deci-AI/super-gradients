import logging
import os
import sys
import time
from typing import Union


from super_gradients.common.environment.env_variables import env_variables


class AutoLoggerConfig:
    """
    A Class for the Automated Logging Config
    """

    filename: Union[str, None]

    def __init__(self):
        self.filename = None

    def _setup_default_logging(self, log_level: str = None) -> None:
        """
        Setup default logging configuration. Usually happens when app starts, and we don't have
        experiment dir yet.
        The default log directory will be `~/sg_logs`
        :param log_level: The default log level to use. If None, uses LOG_LEVEL and CONSOLE_LOG_LEVEL environment vars.
        :return: None
        """

        # There is no _easy_ way to log all events to a single file, when using DDP or DataLoader with num_workers > 1
        # on Windows platform. In both these cases a multiple processes will be spawned and multiple logs may be created.
        # Therefore the log file will have the parent PID to being able to discriminate the logs corresponding to a single run.
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self._setup_logging(
            filename=os.path.expanduser(f"~/sg_logs/logs_{os.getppid()}_{timestamp}.log"),
            copy_already_logged_messages=False,
            filemode="w",
            log_level=log_level,
        )

    def _setup_logging(self, filename: str, copy_already_logged_messages: bool, filemode: str = "a", log_level: str = None) -> None:
        """
        Sets the logging configuration to store messages to specific file
        :param filename: Output log file
        :param filemode: Open mode for file
        :param copy_already_logged_messages: Controls whether messages from previous log configuration should be copied
               to new place. This is helpful to transfer diagnostic messages (from the app start) to experiment dir.
        :param log_level: The default log level to use. If None, uses LOG_LEVEL and CONSOLE_LOG_LEVEL environment vars.
        :return:
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if copy_already_logged_messages and self.filename is not None and os.path.exists(self.filename):
            with open(self.filename, "r", encoding="utf-8") as src:
                with open(filename, "w") as dst:
                    dst.write(src.read())

        file_logging_level = log_level or env_variables.FILE_LOG_LEVEL
        console_logging_level = log_level or env_variables.CONSOLE_LOG_LEVEL

        cur_version = sys.version_info
        python_38 = (3, 8)
        python_39 = (3, 9)
        manager = logging.getLogger("").manager

        extra_kwargs = {}
        if cur_version >= python_38:
            extra_kwargs = dict(
                force=True,
            )
        else:
            # If the logging does not support force=True, we should manually delete handlers
            for h in manager.root.handlers:
                try:
                    h.close()
                except AttributeError:
                    pass
            del manager.root.handlers[:]

        if cur_version >= python_39:
            extra_kwargs["encoding"] = "utf-8"

        logging.basicConfig(
            filename=filename,
            filemode=filemode,
            format="%(asctime)s %(levelname)s - %(name)s - %(message)s",
            datefmt="[%Y-%m-%d %H:%M:%S]",
            level=file_logging_level,
            **extra_kwargs,
        )

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_logging_level)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s - %(filename)s - %(message)s",
                datefmt="[%Y-%m-%d %H:%M:%S]",
            )
        )
        manager.root.handlers.append(console_handler)

        self.filename = filename

    @classmethod
    def get_instance(cls):
        global _super_gradients_logger_config
        if _super_gradients_logger_config is None:
            _super_gradients_logger_config = cls()
            _super_gradients_logger_config._setup_default_logging()

        return _super_gradients_logger_config

    @classmethod
    def get_log_file_path(cls) -> str:
        """
        Return the current log file used to store log messages
        :return: Full path to log file
        """
        self = cls.get_instance()
        return self.filename

    @classmethod
    def setup_logging(cls, filename: str, copy_already_logged_messages: bool, filemode: str = "a", log_level: str = None) -> None:
        self = cls.get_instance()
        self._setup_logging(filename, copy_already_logged_messages, filemode, log_level)


_super_gradients_logger_config = None
