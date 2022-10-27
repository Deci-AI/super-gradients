import os
import sys
import logging
import atexit
import time
from datetime import datetime
import functools


from deci_lab_client.client import DeciPlatformClient

current_time = datetime.today().isoformat()

base_logger = logging.getLogger(__name__)
base_logger.setLevel(logging.INFO)

EXPERIMENT = "new_exp"
ERROR_FILE = f"/home/louis.dupont/PycharmProjects/super-gradients/checkpoints/exc_info_{current_time}.log"


def get_exception_logger():
    exception_logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(ERROR_FILE)
    exception_logger.addHandler(file_handler)
    return exception_logger


def log_exceptions(func):
    print(f"logging {func.__name__}")
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            exception_logger = get_exception_logger()
            exception_logger.exception(f"Exception stored in {ERROR_FILE}")
            raise e
    return wrapped_func


def add_onexception_log():
    print("add_onexception_log")
    platform_client = DeciPlatformClient(api_host="api.development.deci.ai")
    platform_client.login(token=os.getenv("DECI_PLATFORM_TOKEN"))
    def wrap_excepthook_with_logging(excepthook):
        """Add log before the excepthook"""
        def excepthook_with_logging(exc_type, exc_value, exc_traceback):
            print("excepthook_with_logging")
            exception_logger = get_exception_logger()
            exception_logger.error(f"Exception stored in {ERROR_FILE}", exc_info=(exc_type, exc_value, exc_traceback))
            excepthook(exc_type, exc_value, exc_traceback)

            if os.path.isfile(ERROR_FILE):
                platform_client.register_experiment(name=EXPERIMENT)
                print(f"Uploading exception ({ERROR_FILE}) to deci platform ...")
                platform_client.save_experiment_file(file_path=ERROR_FILE)
                time.sleep(10)  # Would be great to have a sync
                print(f"Exception was uploaded to deci platform!")
        return excepthook_with_logging

    sys.excepthook = wrap_excepthook_with_logging(sys.excepthook)
    # sys.excepthook = wrap_excepthook_with_logging(sys.__excepthook__)


def add_onexit_upload():
    # This might not be required if the upload is handled with exception (i.e. before exit)
    print("add_onexit_upload")
    platform_client = DeciPlatformClient(api_host="api.development.deci.ai")
    platform_client.login(token=os.getenv("DECI_PLATFORM_TOKEN"))

    def exit_handler():
        print("BEFORE EXIT")
        if os.path.isfile(ERROR_FILE):
            platform_client.register_experiment(name=EXPERIMENT)
            print(f"Uploading exception ({ERROR_FILE}) to deci platform ...")
            platform_client.save_experiment_file(file_path=ERROR_FILE)
            time.sleep(20)  # Would be great to have a synch
            print(f"Exception was uploaded to deci platform!")

    atexit.register(exit_handler)


def setup_env(token=None):
    os.environ[
        "DECI_PLATFORM_TOKEN"] = "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJsb3Vpcy5kdXBvbnRAZGVjaS5haSIsImNvbXBhbnlfaWQiOiJjNGYyNmI4OS04OGI0LTQ1M2EtYmQ2OC1lYTdmZWFjMTBlOWYiLCJ3b3Jrc3BhY2VfaWQiOiJjNGYyNmI4OS04OGI0LTQ1M2EtYmQ2OC1lYTdmZWFjMTBlOWYiLCJjb21wYW55X25hbWUiOiJkZWNpLmFpLUxvdWlzLUR1cG9udCIsInVzZXJfaWQiOiI2MzJmMTkxNS1mNTBmLTRjM2YtYjdmNS1lYjYxZDMyZWU0NjQiLCJzb3VyY2UiOiJQbGF0Zm9ybSIsImV4cCI6OTEzMTY2NzYzNCwiaXNfcHJlbWl1bSI6ZmFsc2V9.Jm19rCn9F2vOqo8yfqTSxgkTkfH1X1Ct8za5j2r4KvfW3b7_iFPY2hcmBnYxv6FHABL6s2k43OVeLtYQXVsr2Q"
    if os.getenv("DECI_PLATFORM_TOKEN"):
        print("DECI_PLATFORM_TOKEN was specified")
        add_onexception_log()
        # add_onexit_upload()


setup_env()
