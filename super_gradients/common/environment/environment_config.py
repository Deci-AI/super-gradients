import logging
from os import environ

AWS_ENV_NAME = environ.get('ENVIRONMENT_NAME')

AWS_ENVIRONMENTS = ['development', 'staging', 'production']
if AWS_ENV_NAME not in AWS_ENVIRONMENTS:
    if AWS_ENV_NAME is None:
        if AWS_ENV_NAME not in AWS_ENVIRONMENTS:
            print(
                f'You did not mention an AWS environment.'
                f'You can set the environment variable ENVIRONMENT_NAME with one of the values: {",".join(AWS_ENVIRONMENTS)}')
        else:
            print(
                f'Bad AWS environment name: {AWS_ENV_NAME}. Please set an environment variable named ENVIRONMENT_NAME with one of the values: {",".join(AWS_ENVIRONMENTS)}')

# If this flag exists, ElasticSerach handler in every ILogger class (auto-logger) will be disabled.
# For standalone super_gradients.infra clients.
DONT_USE_ELASTICSEARCH_LOGGER = environ.get('DONT_USE_ELASTICSEARCH_LOGGER') is not None

# Controlling the default logging level via environment variable
DEFAULT_LOGGING_LEVEL = environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=DEFAULT_LOGGING_LEVEL)  # Set the default level for all libraries - including 3rd party packages

DDP_LOCAL_RANK = -1
