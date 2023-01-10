from super_gradients.sanity_check.check_packages import check_packages
from super_gradients.sanity_check.check_os import check_os
from super_gradients.sanity_check.display_utils import log_test_msg, display_muting_instructions

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.ddp_utils import is_main_process


logger = get_logger(__name__, "DEBUG")


def run_env_sanity_check():
    """Run the sanity check tests and log everything that does not meet requirements"""

    requirement_checkers = {
        "Operating System": check_os,
        "Installed packaged": check_packages,
    }

    log_test_msg("SuperGradients Sanity Check Started")
    log_test_msg(f"Checking the following components: {list(requirement_checkers.keys())}")
    log_test_msg("________________________")

    success = True
    for test_name, test_function in requirement_checkers.items():
        log_test_msg(f"Verifying {test_name}...")
        test_success = test_function(test_name)
        success = success and test_success
        log_test_msg("________________________")

    if success:
        log_test_msg("Great, Looks like the current environment meet's Deci's requirements!")
    else:
        log_test_msg("The current environment does not meet Deci's needs, see errors above.")

    display_muting_instructions()


def env_sanity_check():
    """Run the sanity check tests and log everything that does not meet requirements"""
    if is_main_process():
        run_env_sanity_check()


if __name__ == "__main__":
    env_sanity_check()
