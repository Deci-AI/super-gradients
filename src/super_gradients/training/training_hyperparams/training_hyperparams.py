import hydra
import pkg_resources
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from super_gradients.training.utils.utils import override_default_params_without_nones
from super_gradients.training.utils.hydra_utils import normalize_path
from super_gradients.common.abstractions.abstract_logger import get_logger
from typing import Dict

logger = get_logger(__name__)


def get(config_name, overriding_params: Dict = None) -> Dict:
    """
    Class for creating training hyper parameters dictionary, taking defaults from yaml
     files in src/super_gradients/recipes.

    :param overriding_params: Dict, dictionary like object containing entries to override in the recipe's training
     hyper parameters dictionary.
    :param config_name: yaml config filename in recipes (for example coco2017_yolox).
    """
    if overriding_params is None:
        overriding_params = dict()
    GlobalHydra.instance().clear()
    sg_recipes_dir = pkg_resources.resource_filename("super_gradients.recipes", "")
    with initialize_config_dir(config_dir=normalize_path(sg_recipes_dir), version_base="1.2"):
        cfg = compose(config_name=normalize_path(config_name))
        cfg = hydra.utils.instantiate(cfg)
        training_params = cfg.training_hyperparams
        training_params = override_default_params_without_nones(overriding_params, training_params)
        return training_params


def cifar10_resnet_train_params(overriding_params: Dict = None):
    return get("cifar10_resnet", overriding_params)


def cityscapes_ddrnet_train_params(overriding_params: Dict = None):
    return get("cityscapes_ddrnet", overriding_params)


def cityscapes_regseg48_train_params(overriding_params: Dict = None):
    return get("cityscapes_regseg48", overriding_params)


def cityscapes_stdc_base_train_params(overriding_params: Dict = None):
    return get("cityscapes_stdc_base", overriding_params)


def cityscapes_stdc_seg50_train_params(overriding_params: Dict = None):
    return get("cityscapes_stdc_seg50", overriding_params)


def cityscapes_stdc_seg75_train_params(overriding_params: Dict = None):
    return get("cityscapes_stdc_seg75", overriding_params)


def coco2017_ssd_lite_mobilenet_v2_train_params(overriding_params: Dict = None):
    return get("coco2017_ssd_lite_mobilenet_v2", overriding_params)


def coco2017_yolox_train_params(overriding_params: Dict = None):
    return get("coco2017_yolox", overriding_params)


def coco_segmentation_shelfnet_lw_train_params(overriding_params: Dict = None):
    return get("coco_segmentation_shelfnet_lw", overriding_params)


def imagenet_efficientnet_train_params(overriding_params: Dict = None):
    return get("imagenet_efficientnet", overriding_params)


def imagenet_mobilenetv2_train_params(overriding_params: Dict = None):
    return get("imagenet_mobilenetv2", overriding_params)


def imagenet_mobilenetv3_base_train_params(overriding_params: Dict = None):
    return get("imagenet_mobilenetv3_base", overriding_params)


def imagenet_mobilenetv3_large_train_params(overriding_params: Dict = None):
    return get("imagenet_mobilenetv3_large", overriding_params)


def imagenet_mobilenetv3_small_train_params(overriding_params: Dict = None):
    return get("imagenet_mobilenetv3_small", overriding_params)


def imagenet_regnetY_train_params(overriding_params: Dict = None):
    return get("imagenet_regnetY", overriding_params)


def imagenet_repvgg_train_params(overriding_params: Dict = None):
    return get("imagenet_repvgg", overriding_params)


def imagenet_resnet50_train_params(overriding_params: Dict = None):
    return get("imagenet_resnet50", overriding_params)


def imagenet_resnet50_kd_train_params(overriding_params: Dict = None):
    return get("imagenet_resnet50_kd", overriding_params)


def imagenet_vit_base_train_params(overriding_params: Dict = None):
    return get("imagenet_vit_base", overriding_params)


def imagenet_vit_large_train_params(overriding_params: Dict = None):
    return get("imagenet_vit_large", overriding_params)
