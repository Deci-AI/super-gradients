import copy
from typing import Union, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor

from super_gradients.common.object_names import Models
from super_gradients.common.registry.registry import register_model
from super_gradients.module_interfaces import ExportableObjectDetectionModel, AbstractObjectDetectionDecodingModule
from super_gradients.training.models.arch_params_factory import get_arch_params
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.utils.detection_utils import convert_cxcywh_bbox_to_xyxy
from super_gradients.training.utils.utils import HpmStruct, get_param

DEFAULT_SSD_MOBILENET_V1_ARCH_PARAMS = get_arch_params("ssd_mobilenetv1_arch_params")
DEFAULT_SSD_LITE_MOBILENET_V2_ARCH_PARAMS = get_arch_params("ssd_lite_mobilenetv2_arch_params")


@register_model(Models.SSD_MOBILENET_V1)
class SSDMobileNetV1(CustomizableDetector, ExportableObjectDetectionModel):
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

    def get_decoding_module(self, num_pre_nms_predictions: int, **kwargs) -> AbstractObjectDetectionDecodingModule:
        return SSDDecodingModule(num_pre_nms_predictions=num_pre_nms_predictions, **kwargs)


@register_model(Models.SSD_LITE_MOBILENET_V2)
class SSDLiteMobileNetV2(CustomizableDetector, ExportableObjectDetectionModel):
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

    def get_decoding_module(self, num_pre_nms_predictions: int, **kwargs) -> AbstractObjectDetectionDecodingModule:
        return SSDDecodingModule(num_pre_nms_predictions=num_pre_nms_predictions, **kwargs)


class SSDDecodingModule(AbstractObjectDetectionDecodingModule):
    def __init__(self, num_pre_nms_predictions: int, with_confidence: bool = True):
        super().__init__()
        self.num_pre_nms_predictions = num_pre_nms_predictions
        self.with_confidence = with_confidence

    def forward(self, inputs) -> Tuple[Tensor, Tensor]:
        predictions = inputs[0]

        cxcywh = predictions[:, :, :4]
        conf = predictions[:, :, 4:5]
        pred_scores = predictions[:, :, 5:]
        pred_bboxes = convert_cxcywh_bbox_to_xyxy(cxcywh)

        if self.with_confidence:
            pred_scores = pred_scores * conf

        pred_cls_conf, _ = torch.max(pred_scores, dim=2)
        nms_top_k = self.num_pre_nms_predictions
        topk_candidates = torch.topk(pred_cls_conf, dim=1, k=nms_top_k, largest=True, sorted=True)

        offsets = nms_top_k * torch.arange(pred_cls_conf.size(0), device=pred_cls_conf.device)
        flat_indices = topk_candidates.indices + offsets.reshape(pred_cls_conf.size(0), 1)
        flat_indices = torch.flatten(flat_indices)

        output_pred_bboxes = pred_bboxes.reshape(-1, pred_bboxes.size(2))[flat_indices, :].reshape(pred_bboxes.size(0), nms_top_k, pred_bboxes.size(2))
        output_pred_scores = pred_scores.reshape(-1, pred_scores.size(2))[flat_indices, :].reshape(pred_scores.size(0), nms_top_k, pred_scores.size(2))

        return output_pred_bboxes, output_pred_scores
