import copy
from typing import Union

from omegaconf import DictConfig

from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.training.models.arch_params_factory import get_arch_params
from super_gradients.training.utils.utils import HpmStruct, get_param
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector


DEFAULT_SSD_MOBILENET_V1_ARCH_PARAMS = get_arch_params("ssd_mobilenetv1_arch_params")
DEFAULT_SSD_LITE_MOBILENET_V2_ARCH_PARAMS = get_arch_params("ssd_lite_mobilenetv2_arch_params")


@register_model(Models.SSD_MOBILENET_V1)
class SSDMobileNetV1(CustomizableDetector):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig], in_channels: int = 3):
        merged_arch_params = HpmStruct(**copy.deepcopy(DEFAULT_SSD_MOBILENET_V1_ARCH_PARAMS))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            neck=merged_arch_params.neck,
            heads=merged_arch_params.heads,
            num_classes=get_param(merged_arch_params, "num_classes", None),
            bn_eps=get_param(merged_arch_params, "bn_eps", None),
            bn_momentum=get_param(merged_arch_params, "bn_momentum", None),
            inplace_act=get_param(merged_arch_params, "inplace_act", True),
            in_channels=in_channels,
        )


@register_model(Models.SSD_LITE_MOBILENET_V2)
class SSDLiteMobileNetV2(CustomizableDetector):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig], in_channels: int = 3):
        merged_arch_params = HpmStruct(**copy.deepcopy(DEFAULT_SSD_LITE_MOBILENET_V2_ARCH_PARAMS))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            neck=merged_arch_params.neck,
            heads=merged_arch_params.heads,
            num_classes=get_param(merged_arch_params, "num_classes", None),
            bn_eps=get_param(merged_arch_params, "bn_eps", None),
            bn_momentum=get_param(merged_arch_params, "bn_momentum", None),
            inplace_act=get_param(merged_arch_params, "inplace_act", True),
            in_channels=in_channels,
        )
