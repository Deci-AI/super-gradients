import argparse
import sys
from typing import Any


EXTRA_ARGS = []


def pop_arg(arg_name: str, default_value: Any = None) -> Any:
    """Get the specified args and remove them from argv"""

    parser = argparse.ArgumentParser()
    parser.add_argument(f"--{arg_name}", default=default_value)
    args, _ = parser.parse_known_args()

    # Remove the ddp args to not have a conflict with the use of hydra
    for val in filter(lambda x: x.startswith(f"--{arg_name}"), sys.argv):
        EXTRA_ARGS.append(val)
        sys.argv.remove(val)
    return vars(args)[arg_name]
