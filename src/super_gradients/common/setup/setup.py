import functools
from typing import Callable, Optional

import hydra
from omegaconf import DictConfig

from super_gradients.training import utils as core_utils
from super_gradients.training.utils.distributed_training_utils import setup_gpu_mode
from super_gradients.common.environment.env_helpers import init_trainer
from super_gradients.common.data_types.enum import MultiGPUMode


def main(config_path: Optional[str] = None, config_name: Optional[str] = None) -> Callable:
    """Decorator that setups your environment to seamlessly work with recipes and/or DDP.

    Includes:
        - Recipe (Hydra) config file parsing
        - Out of the box environment setup for DDP.

    :param config_path:     The config path, a directory relative to the declaring python file
    :param config_name:     The name of the config (usually the file name without the .yaml extension)

    Note that:
        >>> @super_gradients.recipe_main(config_path=...)
        >>> def task_function(cfg: DictConfig) -> None:
        >>>     # Do something

    Is equivalent to: (deprecated)
        >>> init_trainer()
        >>> @hydra.main(config_path=...)
        >>> def main(cfg: DictConfig):
        >>>     setup_gpu_mode(gpu_mode=...)
        >>>     try:
        >>>         task_function(cfg)
        >>>     except Exception as e:
        >>>         exception_handler(e)
    """

    def recipe_main_decorator(task_function: Callable) -> Callable:
        """Decorator that wraps the function with a @hydra.main, but including others steps required to properly setup super_gradients.

        :param task_function: The (main) function that is wrapped. Should include all the SuperGradient code.
        """

        init_trainer()

        @hydra.main(config_path=config_path, config_name=config_name, version_base="1.2")
        @functools.wraps(task_function)  # This is required to be called along with @hydra.main in order to call @hydra.main out of '__main__' scope
        def wrapped_task_function_with_env_setup(cfg: DictConfig) -> None:
            """Wrap the task, adding setup and exception handler.
            :param cfg: The parsed DictConfig from yaml recipe files or a dictionary
            """
            setup_gpu_mode(gpu_mode=core_utils.get_param(cfg, "multi_gpu", MultiGPUMode.OFF), num_gpus=core_utils.get_param(cfg, "num_gpus"))
            try:
                task_function(cfg)
            except Exception as e:
                exception_handler(e)
                raise e

        return wrapped_task_function_with_env_setup

    return recipe_main_decorator


def exception_handler(e: Exception) -> None:
    """Nothing implemented yet, coming soon."""
    pass
