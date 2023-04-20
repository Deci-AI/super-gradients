"""
Example code for running QAT on SuperGradient's recipes.

General use: python train_from_recipe.py --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""

import hydra
import pkg_resources
from omegaconf import DictConfig

from super_gradients import init_trainer
from super_gradients.training.qat_trainer.qat_trainer import QATTrainer


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def _main(cfg: DictConfig) -> None:
    QATTrainer.train_from_config(cfg)


def main():
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    _main()


if __name__ == "__main__":
    main()
