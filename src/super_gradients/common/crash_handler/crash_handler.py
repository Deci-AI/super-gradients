import sys

from super_gradients.common.crash_handler.crash_tips_setup import setup_crash_tips
from super_gradients.common.crash_handler.exception_monitoring_setup import setup_pro_user_monitoring
from super_gradients.common.crash_handler.exception import register_exceptions
from super_gradients.common.environment.env_variables import env_variables


def setup_crash_handler():
    """Setup the environment to handle crashes, with crash tips and more."""
    is_setup_crash_tips = setup_crash_tips()
    is_setup_pro_user_monitoring = setup_pro_user_monitoring()
    if is_setup_crash_tips or is_setup_pro_user_monitoring:  # We don't want to wrap sys.excepthook when not required

        # This prevents hydra.main to catch errors that happen in the decorated function
        # (which leads sys.excepthook to never be called)
        env_variables.HYDRA_FULL_ERROR = "1"

        sys.excepthook = register_exceptions(sys.excepthook)
