import logging
from os import environ



# Controlling the default logging level via environment variable
DEFAULT_LOGGING_LEVEL = environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=DEFAULT_LOGGING_LEVEL
)  # Set the default level for all libraries - including 3rd party packages
