import copy
from typing import Union

from omegaconf import DictConfig

from super_gradients.training.utils.hydra_utils import load_arch_params
from super_gradients.training.utils.utils import HpmStruct
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector


DEFAULT_SSD_MOBILENET_V1_PARAMS = load_arch_params("arch_params", "ssd_mobilenetv1_arch_params.yaml")
DEFAULT_SSD_LITE_MOBILENET_V2_PARAMS = load_arch_params("arch_params", "ssd_lite_mobilenetv2_arch_params.yaml")


class SSDMobileNetV1(CustomizableDetector):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig], in_channels: int = 3):
        merged_arch_params = HpmStruct(**copy.deepcopy(DEFAULT_SSD_MOBILENET_V1_PARAMS))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(merged_arch_params, in_channels=in_channels)


class SSDLiteMobileNetV2(CustomizableDetector):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig], in_channels: int = 3):
        merged_arch_params = HpmStruct(**copy.deepcopy(DEFAULT_SSD_LITE_MOBILENET_V2_PARAMS))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(merged_arch_params, in_channels=in_channels)
