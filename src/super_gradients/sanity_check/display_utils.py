import logging

from super_gradients.common.abstractions.abstract_logger import get_logger


logger = get_logger(__name__, log_level="DEBUG")

# We want the logger to log everything by default
logger.setLevel(logging.DEBUG)

# We give an option to turn off the basic messages
MUTE_VARIABLE_NAME = "DISPLAY_SANITY_CHECK"
display_sanity_check = True  # os.getenv(MUTE_VARIABLE_NAME, "False") == "True"

stdout_log_level = logging.INFO if display_sanity_check else logging.DEBUG


def log_test_error(test_name: str, error: str):
    """Log a test error in the appropriate format"""
    logger.log(logging.ERROR, f"\33[31mFailed to verify {test_name}: {error}\33[0m")


def log_test_msg(msg: str):
    """Log relatively to the value of DISPLAY_SANITY_CHECK"""
    logger.log(stdout_log_level, msg)


def display_muting_instructions():
    """Display instructions on how to mute/unmute the sanity check."""
    if display_sanity_check:
        logger.info(f"** This check can be hidden by setting the env variable {MUTE_VARIABLE_NAME}=False prior to import. **")
    else:
        logger.info("** A sanity check is done when importing super_gradients for the first time. ** ")
        logger.info(f"-> You can see the details by setting the env variable {MUTE_VARIABLE_NAME}=True prior to import.")
