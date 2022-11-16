import sys
import os
import atexit
from typing import Callable

from super_gradients.common.crash_handler.exception import ExceptionInfo
from super_gradients.common.abstractions.abstract_logger import get_logger


logger = get_logger(__name__)


def register_exceptions(excepthook: Callable):
    """Wrap excepthook with a step the saves the exception info to be available in the exit hooks."""

    def excepthook_with_register(exc_type, exc_value, exc_traceback):
        ExceptionInfo.register_exception(exc_type, exc_value, exc_traceback)
        return excepthook(exc_type, exc_value, exc_traceback)

    return excepthook_with_register


def crash_tip_handler():
    """Display a crash tip if an error was raised"""
    crash_tip_message = ExceptionInfo.get_crash_tip_message()
    if crash_tip_message:
        print(crash_tip_message)


def setup_crash_handler():
    """Setup the environment to handle crashes, with crash tips and more."""
    if os.getenv("CRASH_HANDLER", "TRUE") != "False":
        logger.info("Crash tips is enabled. " "You can set your environment variable to CRASH_HANDLER=FALSE to disable it")
        sys.excepthook = register_exceptions(sys.excepthook)
        atexit.register(crash_tip_handler)
    else:
        logger.info("Crash tips is disabled. " "You can set your environment variable to CRASH_HANDLER=TRUE to enable it")
