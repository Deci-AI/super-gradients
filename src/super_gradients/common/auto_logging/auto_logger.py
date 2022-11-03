import logging
import os
import sys
from typing import Union


class AutoLoggerConfig:
    """
    A Class for the Automated Logging Config
    """

    FILE_LOGGING_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
    CONSOLE_LOGGING_LEVEL = os.environ.get("CONSOLE_LOG_LEVEL", "INFO").upper()

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
        self._setup_logging(
            filename=os.path.expanduser(f"~/sg_logs/last_{os.getppid()}.log"),
            copy_already_logged_messages=False,
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

        file_logging_level = log_level or self.FILE_LOGGING_LEVEL
        console_logging_level = log_level or self.CONSOLE_LOGGING_LEVEL

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
        console_handler.setFormatter(logging.Formatter("%(module)s - %(levelname)s - %(message)s"))
        manager.root.handlers.append(console_handler)

        self.filename = filename

    @classmethod
    def getInstance(cls):
        global _sg_logger
        if _sg_logger is None:
            _sg_logger = cls()
            _sg_logger._setup_default_logging()

        return _sg_logger

    @classmethod
    def get_log_file_path(cls) -> str:
        """
        Return the current log file used to store log messages
        :return: Full path to log file
        """
        self = cls.getInstance()
        return self.filename

    @classmethod
    def setup_logging(cls, filename: str, copy_already_logged_messages: bool, filemode: str = "a", log_level: str = None) -> None:
        self = cls.getInstance()
        self._setup_logging(filename, copy_already_logged_messages, filemode, log_level)


_sg_logger = None
