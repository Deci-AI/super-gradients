"""
Example code for running SuperGradient's recipes.

General use: python -m super_gradients.train_from_kd_recipe --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""

from omegaconf import DictConfig
import hydra

from super_gradients import init_trainer
from super_gradients.training.kd_trainer import KDTrainer


@hydra.main(config_path="recipes", version_base="1.2")
def _main(cfg: DictConfig) -> None:
    KDTrainer.train_from_config(cfg)


def main() -> None:
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    _main()


if __name__ == "__main__":
    main()
