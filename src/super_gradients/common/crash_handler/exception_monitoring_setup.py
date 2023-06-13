import os
import logging
import atexit

from super_gradients.common.environment.env_variables import env_variables
from super_gradients.common.environment.ddp_utils import multi_process_safe, is_distributed
from super_gradients.common.crash_handler.exception import ExceptionInfo
from super_gradients.common.auto_logging.console_logging import ConsoleSink
from super_gradients.common.plugins.deci_client import DeciClient, client_enabled

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@multi_process_safe
def exception_upload_handler(platform_client):
    """Upload the log file to the deci platform if an error was raised"""
    # Make sure that the sink is flushed
    ConsoleSink.flush()
    if not is_distributed() and ExceptionInfo.is_exception_raised():

        logger.info("Uploading console log to deci platform ...")
        try:
            platform_client.upload_file_to_s3(tag="SuperGradients", level="error", from_path=ConsoleSink.get_filename())
            logger.info("Exception was uploaded to deci platform!")
        except Exception as e:  # We don't want the code to break at exit because of the client (whatever the reason might be)
            logger.warning(f"Exception could not be uploaded to platform with exception: {e}")


@multi_process_safe
def setup_pro_user_monitoring() -> bool:
    """Setup the pro user environment for error logging and monitoring"""
    if client_enabled:
        if env_variables.UPLOAD_LOGS:
            logger.info("deci-platform-client package detected. activating automatic log uploading")
            logger.info(
                "If you do not have a deci-platform-client credentials or you wish to disable this feature, please set the env variable UPLOAD_LOGS=FALSE"
            )

            # This prevents hydra.main to catch errors that happen in the decorated function
            # (which leads sys.excepthook to never be called)
            os.environ["HYDRA_FULL_ERROR"] = "1"

            logger.info("Connecting to the deci platform ...")
            platform_client = DeciClient()
            logger.info("Connection to the deci platform successful!")

            atexit.register(exception_upload_handler, platform_client)
            return True
        else:
            logger.info("Automatic log upload was disabled. To enable it please set the env variable UPLOAD_LOGS=TRUE")
    return False
