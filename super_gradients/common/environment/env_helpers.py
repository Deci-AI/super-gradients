import argparse
import os
import sys
from functools import wraps

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


def init_trainer():
    """
    a function to initialize the super_gradients environment. This function should be the first thing to be called
    by any code running super_gradients. It resolves conflicts between the different tools, packages and environments used
    and prepares the super_gradients environment.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)  # used by DDP
    args, _ = parser.parse_known_args()

    # remove any flags starting with --local_rank from the argv list
    to_remove = list(filter(lambda x: x.startswith('--local_rank'), sys.argv))
    if len(to_remove) > 0:
        for val in to_remove:
            sys.argv.remove(val)

    environment_config.DDP_LOCAL_RANK = args.local_rank


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
