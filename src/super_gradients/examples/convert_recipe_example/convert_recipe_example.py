"""
Example code for running SuperGradient's recipes.

General use: python convert_recipe_example.py --config-name=DESIRED_RECIPE'S_CONVERSION_PARAMS experiment_name=DESIRED_RECIPE'S_EXPERIMENT_NAME.

For more optoins see : super_gradients/recipes/conversion_params/default_conversion_params.yaml.

Note: conversion_params yaml file should reside under super_gradients/recipes/conversion_params
"""

from omegaconf import DictConfig
import hydra
import pkg_resources
from super_gradients import init_trainer
from super_gradients.training import models


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes.conversion_params", ""), version_base="1.2")
def main(cfg: DictConfig) -> None:
    # INSTANTIATE ALL OBJECTS IN CFG
    models.convert_from_config(cfg)


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
