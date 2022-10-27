"""
Example code for running SuperGradient's recipes.

General use: python train_from_recipe.py --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""

from omegaconf import DictConfig
import hydra
import pkg_resources

from super_gradients import Trainer, init_trainer
from super_gradients.common.upload_feedback import log_exceptions


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
@log_exceptions
def main(cfg: DictConfig) -> None:
    Trainer.train_from_config(cfg)


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
