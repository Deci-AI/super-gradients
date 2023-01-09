import os
import logging
import atexit


from super_gradients.common.environment.ddp_utils import multi_process_safe, is_distributed
from super_gradients.common.crash_handler.exception import ExceptionInfo
from super_gradients.common.auto_logging.console_logging import ConsoleSink

try:
    from deci_lab_client.client import DeciPlatformClient
    from deci_lab_client.types import S3SignedUrl

    _imported_deci_lab_failure = None
except (ImportError, NameError, ModuleNotFoundError) as _import_err:
    _imported_deci_lab_failure = _import_err


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
            data = platform_client.upload_log_url(tag="SuperGradients", level="error")
            signed_url = S3SignedUrl(**data.data)
            platform_client.upload_file_to_s3(from_path=ConsoleSink.get_filename(), s3_signed_url=signed_url)
            logger.info("Exception was uploaded to deci platform!")
        except Exception as e:  # We don't want the code to break at exit because of the client (whatever the reason might be)
            logger.warning(f"Exception could not be uploaded to platform with exception: {e}")


@multi_process_safe
def setup_pro_user_monitoring() -> bool:
    """Setup the pro user environment for error logging and monitoring"""
    if _imported_deci_lab_failure is None:
        upload_console_logs = os.getenv("UPLOAD_LOGS", "TRUE") == "TRUE"
        if upload_console_logs:
            logger.info("deci-lab-client package detected. activating automatic log uploading")
            logger.info("If you do not have a deci-lab-client credentials or you wish to disable this feature, please set the env variable UPLOAD_LOGS=FALSE")

            # This prevents hydra.main to catch errors that happen in the decorated function
            # (which leads sys.excepthook to never be called)
            os.environ["HYDRA_FULL_ERROR"] = "1"

            logger.info("Connecting to the deci platform ...")
            platform_client = DeciPlatformClient()
            platform_client.login(token=os.getenv("DECI_PLATFORM_TOKEN"))
            logger.info("Connection to the deci platform successful!")

            atexit.register(exception_upload_handler, platform_client)
            return True
        else:
            logger.info("Automatic log upload was disabled. To enable it please set the env variable UPLOAD_LOGS=TRUE")
    return False
