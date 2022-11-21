"""
Example code for running SuperGradient's recipes.

General use: python train_from_kd_recipe.py --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""

from omegaconf import DictConfig
import pkg_resources

import super_gradients
from super_gradients.training.kd_trainer import KDTrainer


@super_gradients.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""))
def main(cfg: DictConfig) -> None:
    KDTrainer.train_from_config(cfg)


if __name__ == "__main__":
    main()
