import logging
import os
import sys
from typing import Optional


class AutoLoggerConfig:
    """
    A Class for the Automated Logging Config
    """

    FILE_LOGGING_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
    CONSOLE_LOGGING_LEVEL = os.environ.get("CONSOLE_LOG_LEVEL", "INFO").upper()

    filename: str = None

    @classmethod
    def get_log_file_path(cls) -> str:
        """
        Return the current log file used to store log messages
        :return: Full path to log file
        """
        self = cls.getInstance()
        return self.filename

    @classmethod
    def setup_default_logging(self, log_level: str = None) -> None:
        self.setup_logging(
            filename=os.path.expanduser(f"~/sg_logs/last_{os.getppid()}.log"),
            copy_already_logged_messages=False,
            log_level=log_level,
        )

    @classmethod
    def setup_logging(self, filename: str, copy_already_logged_messages: bool, log_level: str = None) -> None:
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
            del manager.root.handlers

        if cur_version >= python_39:
            extra_kwargs["encoding"] = "utf-8"

        logging.basicConfig(
            filename=filename,
            filemode="a",
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
            _sg_logger.setup_default_logging()

        return _sg_logger


_sg_logger: Optional[AutoLoggerConfig] = None
