import sys
from .display_utils import log_test_error


def check_os(test_name: str) -> bool:
    """Check the operating system name and platform

    :param test_name: Name that is used to refer to this test.
    :return: True if test was successful, False otherwise
    """

    if "linux" not in sys.platform.lower():
        error = "Deci officially supports only Linux kernels. Some features may not work as expected."
        log_test_error(test_name=test_name, error=error)
        return False
    return True
