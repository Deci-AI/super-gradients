import logging
import os
from os import environ

import pkg_resources

try:
    PKG_CHECKPOINTS_DIR = pkg_resources.resource_filename("checkpoints", "")
except Exception:
    os.makedirs(os.path.join(os.getcwd(), "checkpoints"), exist_ok=True)
    PKG_CHECKPOINTS_DIR = os.path.join(os.getcwd(), "checkpoints")


# Controlling the default logging level via environment variable
DEFAULT_LOGGING_LEVEL = environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=DEFAULT_LOGGING_LEVEL
)  # Set the default level for all libraries - including 3rd party packages

DDP_LOCAL_RANK = -1

INIT_TRAINER = False
