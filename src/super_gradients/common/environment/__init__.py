"""
This module is in charge of environment variables and consts.
"""
from super_gradients.common.environment.environment_config import DDP_LOCAL_RANK
from super_gradients.common.environment.env_helpers import init_trainer, is_distributed

__all__ = ['DDP_LOCAL_RANK', 'init_trainer', 'is_distributed']
