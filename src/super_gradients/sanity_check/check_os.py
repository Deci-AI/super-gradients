import sys

from .display_utils import format_error_msg

from super_gradients.common.abstractions.abstract_logger import get_logger


logger = get_logger(__name__, "DEBUG")


def check_os(test_name: str):
    """Check the operating system name and platform

    :param test_name: Name that is used to refer to this test.
    """

    if "linux" not in sys.platform.lower():
        error = "Deci officially supports only Linux kernels. Some features may not work as expected."
        logger.error(msg=format_error_msg(test_name=test_name, error_msg=error))
