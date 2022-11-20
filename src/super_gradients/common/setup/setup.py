import functools
from typing import Callable

import hydra
from omegaconf import DictConfig

from super_gradients.training import utils as core_utils
from super_gradients.training.utils.distributed_training_utils import setup_gpu_mode
from super_gradients.common.environment.env_helpers import init_trainer
from super_gradients.common.data_types.enum import MultiGPUMode


def main(gpu_mode: MultiGPUMode = MultiGPUMode.OFF, num_gpus: int = 0) -> Callable:
    """Decorator to use SuperGradient in the most simple way, including out of the box DDP setup.
    :param gpu_mode:    DDP, DP or Off
    :param num_gpus:    Number of GPU's to use.

    Note that:
        >>> @super_gradients.main(...)
        >>> def main(cfg: DictConfig) -> None:
        >>>     task_function(cfg)

    Is equivalent to:
        >>> init_trainer()
        >>> setup_gpu_mode(gpu_mode=...)
        >>> try:
        >>>     task_function(cfg)
        >>> except Exception as e:
        >>>     exception_handler(e)
    """

    def main_decorator(task_function: Callable) -> Callable:
        """Decorator that wraps the function, including others steps required to properly setup super_gradients.

        :param task_function: The (main) function that is wrapped. Should include all the SuperGradient code.
        """
        init_trainer()

        def task_function_with_setup(*args, **kwargs) -> None:
            """Wrap the task, adding setup and exception handler."""
            setup_gpu_mode(gpu_mode=gpu_mode, num_gpus=num_gpus)
            try:
                task_function(*args, **kwargs)
            except Exception as e:
                exception_handler(e)
                raise e

        return task_function_with_setup

    return main_decorator


def hydra_main(config_path: str, config_name: str = None) -> Callable:
    """Decorator to use SuperGradient in the most simple way, including hydra config file parsing, and out of the box DDP setup

    Build a decorator that wraps the function with a @hydra.main, but including others steps required to properly setup super_gradients.
    :param config_path:     The config path, a directory relative to the declaring python file.
    :param config_name:     The name of the config (usually the file name without the .yaml extension)

    Note that:
        >>> @super_gradients.recipe_main(config_path=...)
        >>> def main(cfg: DictConfig) -> None:
        >>>     task_function(cfg)

    Is equivalent to:
        >>> init_trainer()
        >>> @hydra.main(config_path=...)
        >>> def task_function(cfg: DictConfig):
        >>>     setup_gpu_mode(gpu_mode=cfg.gpu_mode, num_gpus=cfg.num_gpus)
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

        @hydra.main(config_path=config_path, config_name=config_name)
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


def exception_handler(e: Exception):
    """Nothing implemented yet, coming soon."""
    pass
