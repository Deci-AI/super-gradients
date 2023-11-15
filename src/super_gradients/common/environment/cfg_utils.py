import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Mapping

import hydra
import pkg_resources

from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict, DictConfig
from torch.utils.data import DataLoader

from super_gradients.common.environment.omegaconf_utils import register_hydra_resolvers
from super_gradients.common.environment.path_utils import normalize_path
from super_gradients.common.environment.checkpoints_dir_utils import get_checkpoints_dir_path
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


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


def load_experiment_cfg(experiment_name: str, ckpt_root_dir: Optional[str] = None, run_id: Optional[str] = None) -> DictConfig:
    """
    Load the hydra config associated to a specific experiment.

    Background Information: every time an experiment is launched based on a recipe, all the hydra config params are stored in a hidden folder ".hydra".
    This hidden folder is used here to recreate the exact same config as the one that was used to launch the experiment (Also include hydra overrides).

    The motivation is to be able to resume or evaluate an experiment with the exact same config as the one that was used when the experiment was
    initially started, regardless of any change that might have been introduced to the recipe, and also while using the same overrides that were used
    for that experiment.

    :param experiment_name:     Name of the experiment to resume
    :param ckpt_root_dir:       Directory including the checkpoints
    :param run_id:              Optional. Run id of the experiment. If None, the most recent run will be loaded.
    :return:                    The config that was used for that experiment
    """
    if not experiment_name:
        raise ValueError(f"experiment_name should be non empty string but got :{experiment_name}")

    checkpoints_dir_path = Path(get_checkpoints_dir_path(ckpt_root_dir=ckpt_root_dir, experiment_name=experiment_name, run_id=run_id))
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


def export_recipe(config_name: str, save_path: str, config_dir: str = pkg_resources.resource_filename("super_gradients.recipes", "")):
    """
    saves a complete (i.e no inheritance from other yaml configuration files),
     .yaml file that can be ran on its own without the need to keep other configurations which the original
      file inherits from.

    :param config_name: The .yaml config filename (can leave the .yaml postfix out, but not mandatory).

    :param save_path: The config directory path, as absolute file system path.
        When None, will use SG's recipe directory (i.e path/to/super_gradients/recipes)

    :param config_dir: The config directory path, as absolute file system path.
        When None, will use SG's recipe directory (i.e path/to/super_gradients/recipes)

    """
    # NEED TO REGISTER RESOLVERS FIRST
    register_hydra_resolvers()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=normalize_path(config_dir), version_base="1.2"):
        cfg = compose(config_name=config_name)
        OmegaConf.save(config=cfg, f=save_path)
        logger.info(f"Successfully saved recipe at {save_path}. \n" f"Recipe content:\n {cfg}")


def maybe_instantiate_test_loaders(cfg) -> Optional[Mapping[str, DataLoader]]:
    """
    Instantiate test loaders if they are defined in the config.

    :param cfg: Recipe config
    :return:    A mapping from dataset name to test loader or None if no test loaders are defined.
    """
    from super_gradients.training.utils.utils import get_param
    from super_gradients.training import dataloaders

    test_loaders = None
    if "test_dataset_params" in cfg.dataset_params:
        test_dataloaders = get_param(cfg, "test_dataloaders")
        test_dataset_params = cfg.dataset_params.test_dataset_params
        test_dataloader_params = get_param(cfg.dataset_params, "test_dataloader_params")

        if test_dataloaders is not None:
            if not isinstance(test_dataloaders, Mapping):
                raise ValueError("`test_dataloaders` should be a mapping from test_loader_name to test_loader_params.")

            if test_dataloader_params is not None and test_dataloader_params.keys() != test_dataset_params.keys():
                raise ValueError("test_dataloader_params and test_dataset_params should have the same keys.")

        test_loaders = {}
        for dataset_name, dataset_params in test_dataset_params.items():
            loader_name = test_dataloaders[dataset_name] if test_dataloaders is not None else None
            dataset_params = test_dataset_params[dataset_name]
            dataloader_params = test_dataloader_params[dataset_name] if test_dataloader_params is not None else cfg.dataset_params.val_dataloader_params
            loader = dataloaders.get(loader_name, dataset_params=dataset_params, dataloader_params=dataloader_params)
            test_loaders[dataset_name] = loader

    return test_loaders
