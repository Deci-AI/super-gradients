from typing import Dict

import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize_config_dir
import pkg_resources


def get_arch_params(config_name, overriding_params: Dict = None) -> Dict:
    """
    Class for creating arch parameters dictionary, taking defaults from yaml
     files in src/super_gradients/recipes/arch_params.

    :param overriding_params: Dict, dictionary like object containing entries to override.
    :param config_name: arch_params yaml config filename in recipes (for example unet_default_arch_params).
    """
    if overriding_params is None:
        overriding_params = dict()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=pkg_resources.resource_filename("super_gradients.recipes", "arch_params/"), version_base="1.2"):
        cfg = compose(config_name=config_name)
        arch_params = hydra.utils.instantiate(cfg)
        arch_params.update(**overriding_params)
        return arch_params
