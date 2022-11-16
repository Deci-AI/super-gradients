"""
Example code for running SuperGradient's recipes.

General use: python train_from_recipe.py --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""

from omegaconf import DictConfig
import hydra
import pkg_resources

from super_gradients import Trainer, init_trainer
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def main(cfg: DictConfig) -> None:
    try:
        Trainer.train_from_config(cfg)
    except Exception as err:
        logger.exception("Encountered exception during training: {}".format(err))


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
