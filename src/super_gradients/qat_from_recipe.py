"""
Example code for running QAT on SuperGradient's recipes.

General use: python -m super_gradients.qat_from_recipe --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""

import hydra
from omegaconf import DictConfig

from super_gradients import init_trainer, Trainer


@hydra.main(config_path="recipes", version_base="1.2")
def _main(cfg: DictConfig) -> None:
    Trainer.quantize_from_config(cfg)


def main():
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    _main()


if __name__ == "__main__":
    main()
