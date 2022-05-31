"""
Example code for running SuperGradient's recipes.

General use: python train_from_recipe.py --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""

import super_gradients
from omegaconf import DictConfig
import hydra
import pkg_resources
from super_gradients.training.trainer import Trainer


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""))
def main(cfg: DictConfig) -> None:
    Trainer.train(cfg)


if __name__ == "__main__":
    super_gradients.init_trainer()
    main()
