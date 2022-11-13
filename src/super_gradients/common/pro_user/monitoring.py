import os
import sys
import logging
import atexit
import time
from datetime import datetime
from typing import Callable

from super_gradients.common.auto_logging.console_logging import ConsoleSink
from super_gradients.common.environment.env_helpers import multi_process_safe
from super_gradients.common.sg_loggers import BaseSGLogger

try:
    from deci_lab_client.client import DeciPlatformClient

    _imported_deci_lab_failure = None
except (ImportError, NameError, ModuleNotFoundError) as _import_err:
    _imported_deci_lab_failure = _import_err


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExceptionInfo:
    """Holds information about the session exception (if any)"""

    _is_exception_raised = False

    @staticmethod
    def register_exception():
        """Register the exception information into the class"""
        ExceptionInfo._is_exception_raised = True

    @staticmethod
    def is_exception_raised():
        """Check if an exception was raised in the current process"""
        return ExceptionInfo._is_exception_raised


def register_exceptions(excepthook: Callable):
    """Wrap excepthook with a step the saves the exception info to be available in the exit hooks."""

    def excepthook_with_register(exc_type, exc_value, exc_traceback):
        ExceptionInfo.register_exception()
        return excepthook(exc_type, exc_value, exc_traceback)

    return excepthook_with_register


@multi_process_safe
def exception_upload_handler(platform_client):
    """Upload the log file to the deci platform if an error was raised"""
    # Make sure that the sink is flushed
    ConsoleSink.flush()
    if ExceptionInfo.is_exception_raised():

        if platform_client.experiment is None:
            exp_name = BaseSGLogger.get_experiment_name()
            exp_name = exp_name if exp_name is not None else f"experiment_{datetime.today().isoformat()}"
            platform_client.register_experiment(name=exp_name)

        logger.info(f"Uploading log ({ConsoleSink.get_filename()}) to deci platform ...")
        platform_client.save_experiment_file(file_path=ConsoleSink.get_filename())
        platform_client.send_support_logs(log="An error was raised", tag="SuperGradient", level="error")
        time.sleep(10)
        logger.info("Exception was uploaded to deci platform!")


def setup_pro_user_monitoring():
    """Setup the pro user environment for error logging and monitoring"""
    upload_console_logs = os.getenv("UPLOAD_LOGS", "TRUE") == "TRUE"
    if upload_console_logs:
        logger.info(
            "As a pro user, your logs will be uploaded to the deci platform. "
            "If you want to block this behavior, please set the env variable UPLOAD_LOGS=FALSE"
        )

        # This prevents hydra.main to catch errors that happen in the decorated function
        # (which leads sys.excepthook to never be called)
        os.environ["HYDRA_FULL_ERROR"] = "1"

        platform_client = DeciPlatformClient(api_host="api.development.deci.ai")  # TODO: remove api_host="api.development.deci.ai"
        platform_client.login(token=os.getenv("DECI_PLATFORM_TOKEN"))

        sys.excepthook = register_exceptions(sys.excepthook)
        atexit.register(exception_upload_handler, platform_client)
    else:
        logger.info("You disabled the upload of your logs to deci platform." "You can reactivate it by setting the env variable UPLOAD_LOGS=TRUE")


def setup_user_env():
    if _imported_deci_lab_failure is None:
        setup_pro_user_monitoring()
