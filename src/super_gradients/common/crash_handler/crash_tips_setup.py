import atexit

from super_gradients.common.environment.env_variables import env_variables
from super_gradients.common.crash_handler.exception import ExceptionInfo
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


def crash_tip_handler():
    """Display a crash tip if an error was raised"""
    crash_tip_message = ExceptionInfo.get_crash_tip_message()
    if crash_tip_message:
        print(crash_tip_message)


def setup_crash_tips() -> bool:
    if env_variables.CRASH_HANDLER != "FALSE":
        logger.info("Crash tips is enabled. You can set your environment variable to CRASH_HANDLER=FALSE to disable it")
        atexit.register(crash_tip_handler)
        return True
    else:
        logger.info("Crash tips is disabled. You can set your environment variable to CRASH_HANDLER=TRUE to enable it")
        return False
