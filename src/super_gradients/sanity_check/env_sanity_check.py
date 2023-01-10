from super_gradients.sanity_check.check_packages import check_packages
from super_gradients.sanity_check.check_os import check_os

from super_gradients.common.environment.ddp_utils import is_main_process


def env_sanity_check():
    """Run the sanity check tests and log everything that does not meet requirements"""
    if is_main_process():
        check_os(test_name="Operating System")
        check_packages(test_name="Installed packaged")


if __name__ == "__main__":
    env_sanity_check()
