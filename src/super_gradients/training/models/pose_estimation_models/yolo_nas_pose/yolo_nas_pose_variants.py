import copy
from typing import Union, Optional, List, Tuple

import numpy as np
from omegaconf import DictConfig

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.processing_factory import ProcessingFactory
from super_gradients.common.object_names import Models
from super_gradients.common.registry import register_model
from super_gradients.training.models.arch_params_factory import get_arch_params
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.processing.processing import Processing
from super_gradients.training.utils import get_param
from super_gradients.training.utils.utils import HpmStruct

from .yolo_nas_pose_post_prediction_callback import YoloNASPosePostPredictionCallback

logger = get_logger(__name__)


class YoloNASPose(CustomizableDetector):
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
        self._edge_links = None
        self._edge_colors = None
        self._keypoint_colors = None
        self._image_processor = None
        self._default_nms_conf = None

    @classmethod
    def get_post_prediction_callback(cls, conf: float, iou: float) -> YoloNASPosePostPredictionCallback:
        return YoloNASPosePostPredictionCallback(score_threshold=conf, nms_threshold=iou, nms_top_k=1000, max_predictions=300)

    def get_preprocessing_callback(self, **kwargs):
        processing = self.get_processing_params()
        preprocessing_module = processing.get_equivalent_photometric_module()
        return preprocessing_module

    @resolve_param("image_processor", ProcessingFactory())
    def set_dataset_processing_params(
        self,
        edge_links: Union[np.ndarray, List[Tuple[int, int]]],
        edge_colors: Union[np.ndarray, List[Tuple[int, int, int]]],
        keypoint_colors: Union[np.ndarray, List[Tuple[int, int, int]]],
        image_processor: Optional[Processing] = None,
        conf: Optional[float] = None,
    ) -> None:
        """Set the processing parameters for the dataset.

        :param image_processor: (Optional) Image processing objects to reproduce the dataset preprocessing used for training.
        :param conf:            (Optional) Below the confidence threshold, prediction are discarded
        """
        self._edge_links = edge_links or self._edge_links
        self._edge_colors = edge_colors or self._edge_colors
        self._keypoint_colors = keypoint_colors or self._keypoint_colors
        self._image_processor = image_processor or self._image_processor
        self._default_nms_conf = conf or self._default_nms_conf


@register_model(Models.YOLO_NAS_POSE_S)
class YoloNASPose_S(YoloNASPose):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("yolo_nas_pose_s_arch_params")
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

    @staticmethod
    def get_post_prediction_callback(conf: float, iou: float) -> YoloNASPosePostPredictionCallback:
        return YoloNASPosePostPredictionCallback(score_threshold=conf, nms_threshold=iou, nms_top_k=1000, max_predictions=300)

    @property
    def num_classes(self):
        return self.heads.num_classes


@register_model(Models.YOLO_NAS_POSE_M)
class YoloNASPose_M(YoloNASPose):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("yolo_nas_pose_m_arch_params")
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

    @staticmethod
    def get_post_prediction_callback(conf: float, iou: float) -> YoloNASPosePostPredictionCallback:
        return YoloNASPosePostPredictionCallback(score_threshold=conf, nms_threshold=iou, nms_top_k=1000, max_predictions=300)

    @property
    def num_classes(self):
        return self.heads.num_classes


@register_model(Models.YOLO_NAS_POSE_L)
class YoloNASPose_L(YoloNASPose):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("yolo_nas_pose_l_arch_params")
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

    @staticmethod
    def get_post_prediction_callback(conf: float, iou: float) -> YoloNASPosePostPredictionCallback:
        return YoloNASPosePostPredictionCallback(score_threshold=conf, nms_threshold=iou, nms_top_k=1000, max_predictions=300)

    @property
    def num_classes(self):
        return self.heads.num_classes
