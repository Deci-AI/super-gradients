from typing import Dict, Optional

import hydra.utils
from omegaconf import DictConfig

from super_gradients.common.environment.cfg_utils import load_arch_params


def get_arch_params(config_name: str, overriding_params: Dict = None, recipes_dir_path: Optional[str] = None) -> DictConfig:
    """
    Class for creating arch parameters dictionary, taking defaults from yaml
     files in src/super_gradients/recipes/arch_params.

    :param config_name:         Name of the yaml to load (e.g. "resnet18_cifar_arch_params")
    :param overriding_params: Dict, dictionary like object containing entries to override.
    :param recipes_dir_path:    Optional. Main directory where every recipe are stored. (e.g. ../super_gradients/recipes)
                                This directory should include a "arch_params" folder,
                                which itself should include the config file named after config_name.
    """
    overriding_params = overriding_params if overriding_params else dict()

    arch_params = load_arch_params(config_name=config_name, recipes_dir_path=recipes_dir_path)
    arch_params = hydra.utils.instantiate(arch_params)

    arch_params.update(**overriding_params)

    return arch_params
