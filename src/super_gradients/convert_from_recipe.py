"""
Entry point for converting recipe file to self-contained train.py file.

General use: python -m super_gradients.convert_from_recipe --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""

import hydra
from omegaconf import DictConfig

from super_gradients import init_trainer


def convert_from_recipe(cfg: DictConfig, output_script_path: str):
    content = []

    with open(output_script_path, "w") as f:
        f.writelines(content)


@hydra.main(config_path="recipes", version_base="1.2")
def _main(cfg: DictConfig) -> None:
    return convert_from_recipe(cfg, output_script_path="exported_train.py")


def main() -> None:
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    _main()


if __name__ == "__main__":
    main()
