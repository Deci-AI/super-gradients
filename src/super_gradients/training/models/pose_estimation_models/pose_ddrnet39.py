import copy
from typing import Union

from omegaconf import DictConfig

from super_gradients.common.object_names import Models
from super_gradients.common.registry.registry import register_model
from super_gradients.training.models.arch_params_factory import get_arch_params
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.utils import HpmStruct

POSE_DDRNET39_ARCH_PARAMS = get_arch_params("pose_ddrnet39_arch_params")


@register_model(Models.POSE_DDRNET_39)
class PoseDDRNet39(CustomizableDetector):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig], in_channels: int = 3):
        merged_arch_params = HpmStruct(**copy.deepcopy(POSE_DDRNET39_ARCH_PARAMS))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            heads=merged_arch_params.heads,
            num_classes=merged_arch_params.num_classes,
            bn_eps=merged_arch_params.bn_eps,
            bn_momentum=merged_arch_params.bn_momentum,
            inplace_act=merged_arch_params.inplace_act,
            in_channels=in_channels,
        )
