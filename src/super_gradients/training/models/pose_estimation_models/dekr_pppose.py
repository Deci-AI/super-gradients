import copy
from typing import Union

from omegaconf import DictConfig

from super_gradients.training.models.arch_params_factory import get_arch_params
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.utils import HpmStruct

DEKR_PPPOSE_L_ARCH_PARAMS = get_arch_params("dekr_pppose_l_arch_params")


class DEKRPPPoseL(CustomizableDetector):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig], in_channels: int = 3):
        merged_arch_params = HpmStruct(**copy.deepcopy(DEKR_PPPOSE_L_ARCH_PARAMS))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(merged_arch_params, in_channels=in_channels)
