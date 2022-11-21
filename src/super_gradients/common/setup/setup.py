import functools
from typing import Callable, Optional

import hydra
from omegaconf import DictConfig

from super_gradients.training import utils as core_utils
from super_gradients.training.utils.distributed_training_utils import setup_gpu_mode
from super_gradients.common.environment.env_helpers import init_trainer
from super_gradients.common.data_types.enum import MultiGPUMode


# # MIGHT BE DROPED IF WE CHOSE TO WORK WITH SINGLE DECORATOR
# def main(gpu_mode: MultiGPUMode = MultiGPUMode.OFF, num_gpus: int = 0) -> Callable:
#     """Decorator to use SuperGradient in the most simple way, including out of the box DDP setup.
#     :param gpu_mode:    DDP, DP or Off
#     :param num_gpus:    Number of GPU's to use.
#
#     Note that:
#         >>> @super_gradients.main(...)
#         >>> def main(cfg: DictConfig) -> None:
#         >>>     task_function(cfg)
#
#     Is equivalent to:
#         >>> init_trainer()
#         >>> setup_gpu_mode(gpu_mode=...)
#         >>> try:
#         >>>     task_function(cfg)
#         >>> except Exception as e:
#         >>>     exception_handler(e)
#     """
#
#     def main_decorator(task_function: Callable) -> Callable:
#         """Decorator that wraps the function, including others steps required to properly setup super_gradients.
#
#         :param task_function: The (main) function that is wrapped. Should include all the SuperGradient code.
#         """
#         init_trainer()
#
#         def task_function_with_setup(*args, **kwargs) -> None:
#             """Wrap the task, adding setup and exception handler."""
#             setup_gpu_mode(gpu_mode=gpu_mode, num_gpus=num_gpus)
#             try:
#                 task_function(*args, **kwargs)
#             except Exception as e:
#                 exception_handler(e)
#                 raise e
#
#         return task_function_with_setup
#
#     return main_decorator
#
# # MIGHT BE DROPED IF WE CHOSE TO WORK WITH SINGLE DECORATOR
# def hydra_main(config_path: str, config_name: str = None) -> Callable:
#     """Decorator to use SuperGradient in the most simple way, including hydra config file parsing, and out of the box DDP setup
#
#     Build a decorator that wraps the function with a @hydra.main, but including others steps required to properly setup super_gradients.
#     :param config_path:     The config path, a directory relative to the declaring python file.
#     :param config_name:     The name of the config (usually the file name without the .yaml extension)
#
#     Note that:
#         >>> @super_gradients.recipe_main(config_path=...)
#         >>> def main(cfg: DictConfig) -> None:
#         >>>     task_function(cfg)
#
#     Is equivalent to:
#         >>> init_trainer()
#         >>> @hydra.main(config_path=...)
#         >>> def task_function(cfg: DictConfig):
#         >>>     setup_gpu_mode(gpu_mode=cfg.gpu_mode, num_gpus=cfg.num_gpus)
#         >>>     try:
#         >>>         task_function(cfg)
#         >>>     except Exception as e:
#         >>>         exception_handler(e)
#     """
#
#     def recipe_main_decorator(task_function: Callable) -> Callable:
#         """Decorator that wraps the function with a @hydra.main, but including others steps required to properly setup super_gradients.
#
#         :param task_function: The (main) function that is wrapped. Should include all the SuperGradient code.
#         """
#
#         init_trainer()
#
#         @hydra.main(config_path=config_path, config_name=config_name)
#         @functools.wraps(task_function)  # This is required to be called along with @hydra.main in order to call @hydra.main out of '__main__' scope
#         def wrapped_task_function_with_env_setup(cfg: DictConfig) -> None:
#             """Wrap the task, adding setup and exception handler.
#             :param cfg: The parsed DictConfig from yaml recipe files or a dictionary
#             """
#             setup_gpu_mode(gpu_mode=core_utils.get_param(cfg, "multi_gpu", MultiGPUMode.OFF), num_gpus=core_utils.get_param(cfg, "num_gpus"))
#             try:
#                 task_function(cfg)
#             except Exception as e:
#                 exception_handler(e)
#                 raise e
#
#         return wrapped_task_function_with_env_setup
#
#     return recipe_main_decorator


def main(
    config_path: Optional[str] = None, config_name: Optional[str] = None, gpu_mode: Optional[MultiGPUMode] = None, num_gpus: Optional[int] = None
) -> Callable:
    """Decorator that setups your environment to seamlessly work with recipes and/or DDP.

    Includes:
        - Recipe (Hydra) config file parsing
        - Out of the box environment setup for DDP.

    Use cases (mutually exclusive):
    1. You work WITH A RECIPE, in which case your environment will be setup using that recipe only
        :param config_path:     The config path, a directory relative to the declaring python file
        :param config_name:     The name of the config (usually the file name without the .yaml extension)

    2. You work WITHOUT, in which case you need to provide some parameters to setup your environment.
        :param gpu_mode:        DDP, DP or Off
        :param num_gpus:        Number of GPU's to use

    Note that:
        >>> @super_gradients.recipe_main(config_path=...)
        >>> def main(cfg: DictConfig) -> None:
        >>>     task_function(cfg)

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
    use_recipe = config_path or config_name
    if use_recipe and (gpu_mode or num_gpus):
        raise ValueError(
            "(config_path, config_name) and (gpu_mode, num_gpus) are mutually exclusive.\n"
            "You can either work with a recipe (config_path, config_name), "
            "or set the environment yourself (gpu_mode, num_gpus)"
        )

    def recipe_main_decorator(task_function: Callable) -> Callable:

        init_trainer()
        if use_recipe:

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
        else:

            @functools.wraps(task_function)
            def task_function_with_setup(*args, **kwargs) -> None:
                """Wrap the task, adding setup and exception handler."""
                setup_gpu_mode(gpu_mode=gpu_mode, num_gpus=num_gpus)
                try:
                    task_function(*args, **kwargs)
                except Exception as e:
                    exception_handler(e)
                    raise e

            return task_function_with_setup

    return recipe_main_decorator


def exception_handler(e: Exception):
    """Nothing implemented yet, coming soon."""
    pass
