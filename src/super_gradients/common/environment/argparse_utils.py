import argparse
import sys
from typing import Any
from super_gradients.common.abstractions.abstract_logger import get_logger


logger = get_logger(__name__)

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


def pop_local_rank() -> int:
    """Pop the python arg "local-rank". If exists inform the user with a log, otherwise return -1."""
    local_rank = pop_arg("local_rank", default_value=-1)
    if local_rank != -1:
        logger.info("local_rank was automatically parsed from your config.")
    return local_rank
