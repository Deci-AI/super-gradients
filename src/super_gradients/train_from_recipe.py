"""
Entry point for training from a recipe using SuperGradients.

General use: python -m super_gradients.train_from_recipe --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""
import argparse
import sys
from typing import Any

import hydra


def pop_arg(arg_name: str, default_value: Any = None) -> Any:
    """Get the specified args and remove them from argv"""

    parser = argparse.ArgumentParser()
    parser.add_argument(f"--{arg_name}", default=default_value)
    args, _ = parser.parse_known_args()

    # Remove the ddp args to not have a conflict with the use of hydra
    for val in filter(lambda x: x.startswith(f"--{arg_name}"), sys.argv):
        sys.argv.remove(val)
    return vars(args)[arg_name]


@hydra.main(config_path="recipes", version_base="1.2")
def run(config):
    from super_gradients import init_trainer, Trainer

    init_trainer()
    Trainer.train_from_config(config)


if __name__ == "__main__":
    pop_arg("local_rank", default_value=-1)
    run()
