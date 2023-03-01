import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import hydra
import pkg_resources

from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict, DictConfig

from super_gradients.common.environment.path_utils import normalize_path
from super_gradients.common.environment.checkpoints_dir_utils import get_checkpoints_dir_path


class RecipeNotFoundError(Exception):
    def __init__(self, config_name: str, config_dir: str, recipes_dir_path, config_type: str = "", postfix_err_msg: Optional[str] = None):
        config_dir = os.path.abspath(config_dir)
        message = f"Recipe '{os.path.join(config_dir, config_type, config_name.replace('.yaml', ''))}.yaml' was not found.\n"

        if recipes_dir_path is None:
            message += "Note: If you are NOT loading a built-in SuperGradients recipe, please set recipes_dir_path=<path-to-your-recipe-directory>.\n"

        if postfix_err_msg:
            message += postfix_err_msg

        self.config_name = config_name
        self.config_dir = config_dir
        self.recipes_dir_path = recipes_dir_path
        self.message = message
        super().__init__(self.message)


def load_recipe(config_name: str, recipes_dir_path: Optional[str] = None, overrides: Optional[list] = None) -> DictConfig:
    """Load a single a file of the recipe directory.

    :param config_name:         Name of the yaml to load (e.g. "cifar10_resnet")
    :param recipes_dir_path:    Optional. Main directory where every recipe are stored. (e.g. ../super_gradients/recipes)
                                This directory should include a folder corresponding to the subconfig, which itself should
                                include the config file named after config_name.
    :param overrides:           List of hydra overrides for config file
    """
    GlobalHydra.instance().clear()

    config_dir = recipes_dir_path or pkg_resources.resource_filename("super_gradients.recipes", "")

    with initialize_config_dir(config_dir=normalize_path(config_dir), version_base="1.2"):
        try:
            cfg = compose(config_name=normalize_path(config_name), overrides=overrides if overrides else [])
        except hydra.errors.MissingConfigException:
            raise RecipeNotFoundError(config_name=config_name, config_dir=config_dir, recipes_dir_path=recipes_dir_path)
    return cfg


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

    cfg = load_recipe(config_name="config.yaml", recipes_dir_path=normalize_path(str(resume_dir)), overrides=overrides_cfg)
    return cfg


def add_params_to_cfg(cfg: DictConfig, params: List[str]):
    """Add parameters to an existing config

    :param cfg:     OmegaConf config
    :param params:  List of parameters to add, in dotlist format (i.e. ["training_hyperparams.resume=True"])"""
    new_cfg = OmegaConf.from_dotlist(params)
    override_cfg(cfg, new_cfg)


def load_recipe_from_subconfig(config_name: str, config_type: str, recipes_dir_path: Optional[str] = None, overrides: Optional[list] = None) -> DictConfig:
    """Load a single a file (e.g. "resnet18_cifar_arch_params") stored in a subconfig (e.g. "arch_param") of the recipe directory,.

    :param config_name:         Name of the yaml to load (e.g. "resnet18_cifar_arch_params")
    :param config_type:         Type of the subconfig (e.g. "arch_params")
    :param recipes_dir_path:    Optional. Main directory where every recipe are stored. (e.g. ../super_gradients/recipes)
                                This directory should include a folder corresponding to the subconfig,
                                which itself should include the config file named after config_name.
    :param overrides:           List of hydra overrides for config file
    """

    try:
        cfg = load_recipe(config_name=os.path.join(config_type, config_name), recipes_dir_path=recipes_dir_path, overrides=overrides)
    except RecipeNotFoundError as e:
        postfix_err_msg = (
            f"Note: If your recipe is saved at '{os.path.join(e.config_dir, config_name.replace('.yaml', ''))}.yaml', you can load it with load_recipe(...).\n"
        )

        raise RecipeNotFoundError(
            config_name=config_name,
            config_dir=e.config_dir,
            config_type=config_type,
            recipes_dir_path=recipes_dir_path,
            postfix_err_msg=postfix_err_msg,
        )

    # Because of the way we load the subconfig, cfg will start with a single key corresponding to the type (arch_params, ...) and don't want that.
    cfg = cfg[config_type]

    return cfg


def load_arch_params(config_name: str, recipes_dir_path: Optional[str] = None, overrides: Optional[list] = None) -> DictConfig:
    """Load a single arch_params file.
    :param config_name:         Name of the yaml to load (e.g. "resnet18_cifar_arch_params")
    :param recipes_dir_path:    Optional. Main directory where every recipe are stored. (e.g. ../super_gradients/recipes)
                                This directory should include a "arch_params" folder,
                                which itself should include the config file named after config_name.
    :param overrides:           List of hydra overrides for config file
    """
    return load_recipe_from_subconfig(config_name=config_name, recipes_dir_path=recipes_dir_path, overrides=overrides, config_type="arch_params")


def load_training_hyperparams(config_name: str, recipes_dir_path: Optional[str] = None, overrides: Optional[list] = None) -> DictConfig:
    """Load a single training_hyperparams file.
    :param config_name:         Name of the yaml to load (e.g. "cifar10_resnet_train_params")
    :param recipes_dir_path:    Optional. Main directory where every recipe are stored. (e.g. ../super_gradients/recipes)
                                This directory should include a "training_hyperparams" folder,
                                which itself should include the config file named after config_name.
    :param overrides:           List of hydra overrides for config file
    """
    return load_recipe_from_subconfig(config_name=config_name, recipes_dir_path=recipes_dir_path, overrides=overrides, config_type="training_hyperparams")


def load_dataset_params(config_name: str, recipes_dir_path: Optional[str] = None, overrides: Optional[list] = None) -> DictConfig:
    """Load a single dataset_params file.
    :param config_name:         Name of the yaml to load (e.g. "cifar10_dataset_params")
    :param recipes_dir_path:    Optional. Main directory where every recipe are stored. (e.g. ../super_gradients/recipes)
                                This directory should include a "training_hyperparams" folder,
                                which itself should include the config file named after config_name.
    :param overrides:           List of hydra overrides for config file
    """
    return load_recipe_from_subconfig(config_name=config_name, recipes_dir_path=recipes_dir_path, overrides=overrides, config_type="dataset_params")


def override_cfg(cfg: DictConfig, overrides: Union[DictConfig, Dict[str, Any]]) -> None:
    """Override inplace a config with a list of hydra overrides
    :param cfg:         OmegaConf config
    :param overrides:   Dictionary like object that will be used to override cfg
    """
    with open_dict(cfg):  # This is required to add new fields to existing config
        cfg.merge_with(overrides)
