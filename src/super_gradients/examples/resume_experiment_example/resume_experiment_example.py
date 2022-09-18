"""
Example code for running SuperGradient's recipes.

General use: python train_from_recipe.py --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""
from omegaconf import DictConfig
import hydra
import pkg_resources

from super_gradients import Trainer, init_trainer
from super_gradients.common.environment.env_helpers import pop_arg


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def main_recipe(cfg: DictConfig) -> None:
    Trainer.resume_from_recipe(cfg)


def main() -> None:
    experiment_name = pop_arg("experiment_name")
    ckpt_root_dir = pop_arg("ckpt_root_dir")
    Trainer.resume_experiment(experiment_name=experiment_name, ckpt_root_dir=ckpt_root_dir)


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
