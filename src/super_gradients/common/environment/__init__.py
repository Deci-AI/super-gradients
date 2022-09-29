"""
This module is in charge of environment variables and consts.
"""
from super_gradients.common.environment.environment_config import AWS_ENV_NAME
from super_gradients.common.environment.env_helpers import init_trainer, is_distributed

__all__ = ['AWS_ENV_NAME', 'init_trainer', 'is_distributed']
