import os
from pathlib import Path
from typing import List
import pkg_resources

import hydra
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict, DictConfig

from super_gradients.training.utils.checkpoint_utils import get_checkpoints_dir_path


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


def normalize_path(path: str) -> str:
    """Normalize the directory of file path. Replace the Windows-style (\\) path separators with unix ones (/).
    This is necessary when running on Windows since Hydra compose fails to find a configuration file is the config
    directory contains backward slash symbol.

    :param path: Input path string
    :return: Output path string with all \\ symbols replaces with /.
    """
    return path.replace("\\", "/")


def load_arch_params(config_name: str) -> DictConfig:
    """
    :param config_name: name of a yaml with arch parameters
    """
    GlobalHydra.instance().clear()
    sg_recipes_dir = pkg_resources.resource_filename("super_gradients.recipes", "")
    dataset_config = os.path.join("arch_params", config_name)
    with initialize_config_dir(config_dir=normalize_path(sg_recipes_dir), version_base="1.2"):
        # config is relative to a module
        return hydra.utils.instantiate(compose(config_name=normalize_path(dataset_config)).arch_params)
