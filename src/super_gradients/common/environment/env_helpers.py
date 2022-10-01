import argparse
import importlib
import os
import sys
import socket
from functools import wraps
from typing import Any

from omegaconf import OmegaConf

from super_gradients.training.utils.checkpoint_utils import get_checkpoints_dir_path


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


def get_cls(cls_path):
    """
    A resolver for Hydra/OmegaConf to allow getting a class instead on an instance.
    usage:
    class_of_optimizer: ${class:torch.optim.Adam}
    """
    module = '.'.join(cls_path.split('.')[:-1])
    name = cls_path.split('.')[-1]
    importlib.import_module(module)
    return getattr(sys.modules[module], name)


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


def init_trainer():
    """
    Initialize the super_gradients environment.

    This function should be the first thing to be called by any code running super_gradients.
    It resolves conflicts between the different tools, packages and environments used and prepares the super_gradients environment.
    """
    if not os.getenv("IS_TRAINER_INITIALIZED"):

        register_hydra_resolvers()

        # We pop local_rank if it was specified in the args, because it would break
        args_local_rank = pop_arg("local_rank", default_value="-1")

        # If DDP rank is not set yet, set it with the args value.
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(args_local_rank)
        os.environ["IS_TRAINER_INITIALIZED"] = "True"


def register_hydra_resolvers():
    """Register all the hydra resolvers required for the super-gradients recipes."""
    OmegaConf.register_new_resolver("hydra_output_dir", _hydra_output_dir_resolver, replace=True)
    OmegaConf.register_new_resolver("class", lambda *args: get_cls(*args), replace=True)


def _hydra_output_dir_resolver(ckpt_root_dir: str, experiment_name: str) -> str:
    return get_checkpoints_dir_path(experiment_name=experiment_name, ckpt_root_dir=ckpt_root_dir)


def pop_arg(arg_name: str, default_value: Any = None) -> Any:
    """Get the specified args and remove them from argv"""

    parser = argparse.ArgumentParser()
    parser.add_argument(f"--{arg_name}", default=default_value)
    args, _ = parser.parse_known_args()

    # Remove the ddp args to not have a conflict with the use of hydra
    for val in filter(lambda x: x.startswith(f"--{arg_name}"), sys.argv):
        sys.argv.remove(val)
    return vars(args)[arg_name]


def get_ddp_local_rank(default_value: int = -1) -> int:
    return int(os.getenv("LOCAL_RANK", default_value))


def is_distributed() -> bool:
    return get_ddp_local_rank() >= 0


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
        if get_ddp_local_rank() <= 0:
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
        _ip, port = sock.getsockname()
    return port
