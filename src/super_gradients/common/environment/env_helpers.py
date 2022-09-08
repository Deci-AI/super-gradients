import argparse
import os
import sys
import socket
import subprocess
import logging
from functools import wraps
from typing import List

from omegaconf import OmegaConf

from super_gradients.common.environment import environment_config


class TerminalColours:
    """
    Usage: https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python?page=1&tab=votes#tab-top
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ColouredTextFormatter:
    @staticmethod
    def print_coloured_text(text: str, colour: str):
        """
        Prints a text with colour ascii characters.
        """
        return print(''.join([colour, text, TerminalColours.ENDC]))


def get_environ_as_type(environment_variable_name: str, default=None, cast_to_type: type = str) -> object:
    """
    Tries to get an environment variable and cast it into a requested type.
    :return: cast_to_type object, or None if failed.
    :raises ValueError: If the value could not be casted into type 'cast_to_type'
    """
    value = os.environ.get(environment_variable_name, default)
    if value is not None:
        try:
            return cast_to_type(value)
        except Exception as e:
            print(e)
            raise ValueError(
                f'Failed to cast environment variable {environment_variable_name} to type {cast_to_type}: the value {value} is not a valid {cast_to_type}')
    return


def hydra_output_dir_resolver(ckpt_root_dir, experiment_name):
    if ckpt_root_dir is None:
        output_dir_path = (environment_config.PKG_CHECKPOINTS_DIR + os.path.sep + experiment_name)
    else:
        output_dir_path = ckpt_root_dir + os.path.sep + experiment_name
    return output_dir_path


def init_trainer():
    """
    a function to initialize the super_gradients environment. This function should be the first thing to be called
    by any code running super_gradients. It resolves conflicts between the different tools, packages and environments used
    and prepares the super_gradients environment.
    TODO: Rename to setup_env or something more explicit than init_trainer
    """
    register_hydra_resolvers()
    setup_ddp_local_rank()


def setup_ddp_local_rank():
    """Initialize environment_config.DDP_LOCAL_RANK with rank value."""
    # Get local_rank from args if exists.
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)  # used by DDP
    args, _ = parser.parse_known_args()

    # Remove local_rank from args if exists.
    to_remove = list(filter(lambda x: x.startswith('--local_rank'), sys.argv))
    if len(to_remove) > 0:
        for val in to_remove:
            sys.argv.remove(val)

    # Set local_rank with priority order (env variable > args > default value)
    environment_config.DDP_LOCAL_RANK = int(os.getenv("LOCAL_RANK", args.local_rank))


def register_hydra_resolvers():
    """Register all the hydra resolvers required for the super-gradients recipes."""
    OmegaConf.register_new_resolver("hydra_output_dir", hydra_output_dir_resolver, replace=True)


def is_distributed() -> bool:
    return environment_config.DDP_LOCAL_RANK >= 0


def multi_process_safe(func):
    """
    A decorator for making sure a function runs only in main process.
    If not in DDP mode (local_rank = -1), the function will run.
    If in DDP mode, the function will run only in the main process (local_rank = 0)
    This works only for functions with no return value
    """

    def do_nothing(*args, **kwargs):
        pass

    @wraps(func)
    def wrapper(*args, **kwargs):
        if environment_config.DDP_LOCAL_RANK <= 0:
            return func(*args, **kwargs)
        else:
            return do_nothing(*args, **kwargs)

    return wrapper


def find_free_port() -> int:
    """Find an available port of current machine/node.
    Note: there is still a chance the port could be taken by other processes."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Binding to port 0 will cause the OS to find an available port for us
        sock.bind(("", 0))
        _adress, port = sock.getsockname()
    return port
