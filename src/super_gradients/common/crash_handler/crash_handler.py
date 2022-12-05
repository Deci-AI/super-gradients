import sys

from super_gradients.common.crash_handler.crash_tips_setup import setup_crash_tips
from super_gradients.common.crash_handler.exception_monitoring_setup import setup_pro_user_monitoring
from super_gradients.common.crash_handler.exception import register_exceptions


def setup_crash_handler():
    """Setup the environment to handle crashes, with crash tips and more."""
    is_setup_crash_tips = setup_crash_tips()
    is_setup_pro_user_monitoring = setup_pro_user_monitoring()
    if is_setup_crash_tips or is_setup_pro_user_monitoring:  # We don't want to wrap sys.excepthook when not required
        sys.excepthook = register_exceptions(sys.excepthook)
