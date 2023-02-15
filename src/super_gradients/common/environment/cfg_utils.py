import os
from pathlib import Path
from typing import List, Optional
import pkg_resources
from enum import Enum

import hydra
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict, DictConfig

from super_gradients.common.environment.path_utils import normalize_path
from super_gradients.common.environment.checkpoints_dir_utils import get_checkpoints_dir_path


def load_experiment_cfg(experiment_name: str, ckpt_root_dir: str = None) -> DictConfig:
    """
    Load the hydra config associated to a specific experiment.

    Background Information: every time an experiment is launched based on a recipe, all the hydra config params are stored in a hidden folder ".hydra".
    This hidden folder is used here to recreate the exact same config as the one that was used to launch the experiment (Also include hydra overrides).

    The motivation is to be able to resume or evaluate an experiment with the exact same config as the one that was used when the experiment was
    initially started, regardless of any change that might have been introduced to the recipe, and also while using the same overrides that were used
    for that experiment.

    :param experiment_name:     Name of the experiment to resume
    :param ckpt_root_dir:       Directory including the checkpoints
    :return:                    The config that was used for that experiment
    """
    if not experiment_name:
        raise ValueError(f"experiment_name should be non empty string but got :{experiment_name}")

    checkpoints_dir_path = Path(get_checkpoints_dir_path(experiment_name, ckpt_root_dir))
    if not checkpoints_dir_path.exists():
        raise FileNotFoundError(f"Impossible to find checkpoint dir ({checkpoints_dir_path})")

    resume_dir = Path(checkpoints_dir_path) / ".hydra"
    if not resume_dir.exists():
        raise FileNotFoundError(f"The checkpoint directory {checkpoints_dir_path} does not include .hydra artifacts to resume the experiment.")

    # Load overrides that were used in previous run
    overrides_cfg = list(OmegaConf.load(resume_dir / "overrides.yaml"))

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=normalize_path(str(resume_dir)), version_base="1.2"):
        cfg = compose(config_name="config.yaml", overrides=overrides_cfg)
    return cfg


def add_params_to_cfg(cfg: DictConfig, params: List[str]):
    """Add parameters to an existing config

    :param cfg:     OmegaConf config
    :param params:  List of parameters to add, in dotlist format (i.e. ["training_hyperparams.resume=True"])"""
    new_cfg = OmegaConf.from_dotlist(params)
    with open_dict(cfg):  # This is required to add new fields to existing config
        cfg.merge_with(new_cfg)


class ConfigType(Enum):
    MAIN_RECIPE = "recipe"
    ANCHORS = "anchors"
    ARCH_PARAMS = "arch_params"
    CHECKPOINT_PARAMS = "checkpoint_params"
    CONVERSION_PARAMS = "conversion_params"
    DATASET_PARAMS = "dataset_params"
    QUANTIZATION_PARAMS = "quantization_params"
    TRAINING_HYPERPARAMS = "training_hyperparams"


def load_cfg(config_name, config_type: ConfigType, recipes_dir_path: Optional[str] = None) -> DictConfig:
    GlobalHydra.instance().clear()

    if recipes_dir_path is None:
        recipes_dir_path = pkg_resources.resource_filename("super_gradients.recipes", "")

    if config_type == ConfigType.MAIN_RECIPE:
        config_relative_path = config_name
    else:
        config_relative_path = os.path.join(config_type.value, config_name)

    with initialize_config_dir(config_dir=normalize_path(recipes_dir_path), version_base="1.2"):
        # config is relative to a module
        cfg = compose(config_name=normalize_path(config_relative_path))

        if config_type != ConfigType.MAIN_RECIPE:
            cfg = cfg[config_type.value]  # We only want to instantiate and work with a sub set of the whole config (arch_params, ...)

        return hydra.utils.instantiate(cfg)


def load_arch_params(config_name: str, recipes_dir_path: Optional[str] = None) -> DictConfig:
    """
    :param config_name:         Name of a yaml with arch parameters
    :param recipes_dir_path:    Optional. Main directory where every recipe are stored.
                                This directory should include an "arch_params" folder, which itself should include the config file named $config_name.
    """
    return load_cfg(config_name=config_name, recipes_dir_path=recipes_dir_path, config_type=ConfigType.ARCH_PARAMS)


def load_recipe(config_name: str, recipes_dir_path: Optional[str] = None) -> DictConfig:
    """
    :param config_name:         Name of a yaml with arch parameters
    :param recipes_dir_path:    Optional. Main directory where every recipe are stored.
                                This directory should include the config file named $config_name.
    """
    return load_cfg(config_name=config_name, recipes_dir_path=recipes_dir_path, config_type=ConfigType.MAIN_RECIPE)
