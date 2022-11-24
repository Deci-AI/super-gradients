import sys

from super_gradients.common.crash_handler.exception import register_exceptions
from super_gradients.common.crash_handler.crash_tips import setup_crash_tips
from super_gradients.common.crash_handler.exception_monitoring import setup_pro_user_monitoring


def setup_crash_handler():
    """Setup the environment to handle crashes, with crash tips and more."""
    is_setup_crash_tips = setup_crash_tips()
    is_setup_pro_user_monitoring = setup_pro_user_monitoring()
    if is_setup_crash_tips or is_setup_pro_user_monitoring:
        sys.excepthook = register_exceptions(sys.excepthook)
