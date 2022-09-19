"""
Example code for testing SuperGradient's recipes.
NOTE:   This script does NOT run TRAINING,
        so make sure in the recipe that you load a PRETRAINED MODEL
        either from one of your checkpoint or from a pretrained model.

General use: python validate_recipe.py --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""
from omegaconf import DictConfig
import hydra
import pkg_resources

from super_gradients import Trainer, init_trainer


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def main(cfg: DictConfig) -> None:
    Trainer.validate_from_recipe(cfg)


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
