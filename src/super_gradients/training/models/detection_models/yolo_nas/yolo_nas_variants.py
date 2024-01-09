import copy
from typing import Union, Tuple, Optional, Any

import torch
from omegaconf import DictConfig
from torch import Tensor

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.object_names import Models
from super_gradients.common.registry import register_model
from super_gradients.module_interfaces import (
    ExportableObjectDetectionModel,
    AbstractObjectDetectionDecodingModule,
    ModelHasNoPreprocessingParamsException,
    SupportsInputShapeCheck,
)
from super_gradients.training.models.arch_params_factory import get_arch_params
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.utils import get_param
from super_gradients.training.utils.utils import HpmStruct

logger = get_logger(__name__)


class YoloNASDecodingModule(AbstractObjectDetectionDecodingModule):
    __constants__ = ["num_pre_nms_predictions"]

    def __init__(
        self,
        num_pre_nms_predictions: int = 1000,
    ):
        super().__init__()
        self.num_pre_nms_predictions = num_pre_nms_predictions

    @torch.jit.ignore
    def infer_total_number_of_predictions(self, predictions: Any) -> int:
        """

        :param inputs:
        :return:
        """
        if torch.jit.is_tracing():
            pred_bboxes, pred_scores = predictions
        else:
            pred_bboxes, pred_scores = predictions[0]

        return pred_bboxes.size(1)

    def get_num_pre_nms_predictions(self) -> int:
        return self.num_pre_nms_predictions

    def forward(self, inputs: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, ...]]):
        if torch.jit.is_tracing():
            pred_bboxes, pred_scores = inputs
        else:
            pred_bboxes, pred_scores = inputs[0]

        nms_top_k = self.num_pre_nms_predictions
        batch_size, num_anchors, _ = pred_scores.size()

        pred_cls_conf, _ = torch.max(pred_scores, dim=2)  # [B, Anchors]
        topk_candidates = torch.topk(pred_cls_conf, dim=1, k=nms_top_k, largest=True, sorted=True)

        offsets = num_anchors * torch.arange(batch_size, device=pred_cls_conf.device)
        indices_with_offset = topk_candidates.indices + offsets.reshape(batch_size, 1)
        flat_indices = torch.flatten(indices_with_offset)

        output_pred_bboxes = pred_bboxes.reshape(-1, pred_bboxes.size(2))[flat_indices, :].reshape(pred_bboxes.size(0), nms_top_k, pred_bboxes.size(2))
        output_pred_scores = pred_scores.reshape(-1, pred_scores.size(2))[flat_indices, :].reshape(pred_scores.size(0), nms_top_k, pred_scores.size(2))

        return output_pred_bboxes, output_pred_scores


class YoloNAS(ExportableObjectDetectionModel, SupportsInputShapeCheck, CustomizableDetector):
    """

    Export to ONNX/TRT Support matrix
    ONNX files generated with PyTorch 2.0.1 for ONNX opset_version=14

    | Batch Size | Export Engine | Format | OnnxRuntime 1.13.1 | TensorRT 8.4.2 | TensorRT 8.5.3 | TensorRT 8.6.1 |
    |------------|---------------|--------|--------------------|----------------|----------------|----------------|
    | 1          | ONNX          | Flat   | Yes                | Yes            | Yes            | Yes            |
    | >1         | ONNX          | Flat   | Yes                | No             | No             | No             |
    | 1          | ONNX          | Batch  | Yes                | No             | Yes            | Yes            |
    | >1         | ONNX          | Batch  | Yes                | No             | No             | Yes            |
    | 1          | TensorRT      | Flat   | No                 | No             | Yes            | Yes            |
    | >1         | TensorRT      | Flat   | No                 | No             | Yes            | Yes            |
    | 1          | TensorRT      | Batch  | No                 | Yes            | Yes            | Yes            |
    | >1         | TensorRT      | Batch  | No                 | Yes            | Yes            | Yes            |

    """

    def __init__(
        self,
        backbone: Union[str, dict, HpmStruct, DictConfig],
        heads: Union[str, dict, HpmStruct, DictConfig],
        neck: Optional[Union[str, dict, HpmStruct, DictConfig]] = None,
        num_classes: int = None,
        bn_eps: Optional[float] = None,
        bn_momentum: Optional[float] = None,
        inplace_act: Optional[bool] = True,
        in_channels: int = 3,
    ):
        super().__init__(backbone, heads, neck, num_classes, bn_eps, bn_momentum, inplace_act, in_channels)

    def get_post_prediction_callback(
        self, *, conf: float, iou: float, nms_top_k: int, max_predictions: int, multi_label_per_box: bool, class_agnostic_nms: bool
    ) -> PPYoloEPostPredictionCallback:
        """
        Get a post prediction callback for this model.

        :param conf:                A minimum confidence threshold for predictions to be used in post-processing.
        :param iou:                 A IoU threshold for boxes non-maximum suppression.
        :param nms_top_k:           The maximum number of detections to consider for NMS.
        :param max_predictions:     The maximum number of detections to return.
        :param multi_label_per_box: If True, each anchor can produce multiple labels of different classes.
                                    If False, each anchor can produce only one label of the class with the highest score.
        :param class_agnostic_nms:  If True, perform class-agnostic NMS (i.e IoU of boxes of different classes is checked).
                                    If False NMS is performed separately for each class.
        :return:
        """
        return PPYoloEPostPredictionCallback(
            score_threshold=conf,
            nms_threshold=iou,
            nms_top_k=nms_top_k,
            max_predictions=max_predictions,
            multi_label_per_box=multi_label_per_box,
            class_agnostic_nms=class_agnostic_nms,
        )

    def get_decoding_module(self, num_pre_nms_predictions: int, **kwargs) -> AbstractObjectDetectionDecodingModule:
        return YoloNASDecodingModule(num_pre_nms_predictions)

    def get_preprocessing_callback(self, **kwargs):
        processing = self.get_processing_params()
        if processing is None:
            raise ModelHasNoPreprocessingParamsException()
        preprocessing_module = processing.get_equivalent_photometric_module()
        return preprocessing_module

    def get_input_shape_steps(self) -> Tuple[int, int]:
        return 32, 32

    def get_minimum_input_shape_size(self) -> Tuple[int, int]:
        return 32, 32


@register_model(Models.YOLO_NAS_S)
class YoloNAS_S(YoloNAS):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("yolo_nas_s_arch_params")
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            neck=merged_arch_params.neck,
            heads=merged_arch_params.heads,
            num_classes=get_param(merged_arch_params, "num_classes", None),
            in_channels=get_param(merged_arch_params, "in_channels", 3),
            bn_momentum=get_param(merged_arch_params, "bn_momentum", None),
            bn_eps=get_param(merged_arch_params, "bn_eps", None),
            inplace_act=get_param(merged_arch_params, "inplace_act", None),
        )

    @property
    def num_classes(self):
        return self.heads.num_classes


@register_model(Models.YOLO_NAS_M)
class YoloNAS_M(YoloNAS):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("yolo_nas_m_arch_params")
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            neck=merged_arch_params.neck,
            heads=merged_arch_params.heads,
            num_classes=get_param(merged_arch_params, "num_classes", None),
            in_channels=get_param(merged_arch_params, "in_channels", 3),
            bn_momentum=get_param(merged_arch_params, "bn_momentum", None),
            bn_eps=get_param(merged_arch_params, "bn_eps", None),
            inplace_act=get_param(merged_arch_params, "inplace_act", None),
        )

    @property
    def num_classes(self):
        return self.heads.num_classes


@register_model(Models.YOLO_NAS_L)
class YoloNAS_L(YoloNAS):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("yolo_nas_l_arch_params")
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            neck=merged_arch_params.neck,
            heads=merged_arch_params.heads,
            num_classes=get_param(merged_arch_params, "num_classes", None),
            in_channels=get_param(merged_arch_params, "in_channels", 3),
            bn_momentum=get_param(merged_arch_params, "bn_momentum", None),
            bn_eps=get_param(merged_arch_params, "bn_eps", None),
            inplace_act=get_param(merged_arch_params, "inplace_act", None),
        )

    @property
    def num_classes(self):
        return self.heads.num_classes
