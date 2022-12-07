"""
This module is in charge of environment variables and consts.
"""
from super_gradients.common.environment.environment_config import device_config
from super_gradients.common.environment.ddp_utils import init_trainer, is_distributed

__all__ = ["device_config", "init_trainer", "is_distributed"]
