"""
Example code for running SuperGradient's recipes.

General use: python convert_recipe_example.py --config-name="DESIRED_RECIPE'S_CONVERSION_PARAMS".

Note: conversion_params yaml file should reside under super_gradients/recipes/conversion_params
"""

from omegaconf import DictConfig
import hydra
import pkg_resources
from super_gradients.training import models
from super_gradients import init_trainer
from omegaconf import OmegaConf
from super_gradients.training.models.conversion import prepare_conversion_cfgs
from super_gradients.training.utils.sg_trainer_utils import parse_args


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes.conversion_params", ""), version_base="1.2")
def main(cfg: DictConfig) -> None:
    # INSTANTIATE ALL OBJECTS IN CFG
    cfg, experiment_cfg = prepare_conversion_cfgs(cfg)

    # BUILD NETWORK
    model = models.get(
        model_name=experiment_cfg.architecture,
        num_classes=experiment_cfg.arch_params.num_classes,
        arch_params=experiment_cfg.arch_params,
        strict_load=cfg.strict_load,
        checkpoint_path=cfg.checkpoint_path,
    )

    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = parse_args(cfg, models.convert_to_onnx)

    models.convert_to_onnx(model=model, **cfg)


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
