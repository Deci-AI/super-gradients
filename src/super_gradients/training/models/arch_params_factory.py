from typing import Dict

from omegaconf import DictConfig

from super_gradients.training.utils.hydra_utils import load_arch_params


def get_arch_params(config_name: str, overriding_params: Dict = None) -> DictConfig:
    """
    Class for creating arch parameters dictionary, taking defaults from yaml
     files in src/super_gradients/recipes/arch_params.

    :param config_name: arch_params yaml config filename in recipes (for example unet_default_arch_params).
    :param overriding_params: Dict, dictionary like object containing entries to override.
    """
    overriding_params = overriding_params if overriding_params else dict()

    arch_params = load_arch_params(config_name=config_name)
    arch_params.update(**overriding_params)

    return arch_params
