import logging
from os import environ

import pkg_resources

PKG_CHECKPOINTS_DIR = pkg_resources.resource_filename("checkpoints", "")

AWS_ENV_NAME = environ.get("ENVIRONMENT_NAME")

AWS_ENVIRONMENTS = ["development", "staging", "production"]
if AWS_ENV_NAME not in AWS_ENVIRONMENTS:
    if AWS_ENV_NAME is None:
        if AWS_ENV_NAME not in AWS_ENVIRONMENTS:
            print(
                f"You did not mention an AWS environment."
                f'You can set the environment variable ENVIRONMENT_NAME with one of the values: {",".join(AWS_ENVIRONMENTS)}'
            )
        else:
            print(
                f'Bad AWS environment name: {AWS_ENV_NAME}. Please set an environment variable named ENVIRONMENT_NAME '
                f'with one of the values: {",".join(AWS_ENVIRONMENTS)}'
            )


# def set_global_log_level(level: str):
#     """Force the initialization of the logging config according to the log level.
#     Be aware that this will also the logging level of already existing loggers."""
#     logging.basicConfig(level=level, force=True)  # Set the default level for all libraries - including 3rd party packages


# Controlling the default logging level via environment variable
DEFAULT_SUBPROCESS_LOGGING_LEVEL = environ.get("SUBPROCESS_LOG_LEVEL", "ERROR").upper()
DEFAULT_LOGGING_LEVEL = environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=DEFAULT_LOGGING_LEVEL)
# set_global_log_level(DEFAULT_LOGGING_LEVEL)

import os
DDP_LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
