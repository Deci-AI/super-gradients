import copy
from functools import lru_cache
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
from super_gradients.training.pipelines.pipelines import PoseEstimationPipeline
from super_gradients.training.processing.processing import Processing
from super_gradients.training.utils import get_param
from super_gradients.training.utils.media.image import ImageSource
from super_gradients.training.utils.predict import PoseEstimationPrediction
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
        self._default_nms_iou = None
        self._default_pre_nms_max_predictions = None
        self._default_post_nms_max_predictions = None

    def predict(
        self,
        images: ImageSource,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        pre_nms_max_predictions: Optional[int] = None,
        post_nms_max_predictions: Optional[int] = None,
        batch_size: int = 32,
        fuse_model: bool = True,
    ) -> PoseEstimationPrediction:
        """Predict an image or a list of images.

        :param images:      Images to predict.
        :param iou:         (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:        (Optional) Below the confidence threshold, prediction are discarded.
                            If None, the default value associated to the training is used.
        :param batch_size:  Maximum number of images to process at the same time.
        :param fuse_model:  If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        pipeline = self._get_pipeline(
            iou=iou,
            conf=conf,
            pre_nms_max_predictions=pre_nms_max_predictions,
            post_nms_max_predictions=post_nms_max_predictions,
            fuse_model=fuse_model,
        )
        return pipeline(images, batch_size=batch_size)  # type: ignore

    @lru_cache(maxsize=1)
    def _get_pipeline(
        self,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        pre_nms_max_predictions: Optional[int] = None,
        post_nms_max_predictions: Optional[int] = None,
        fuse_model: bool = True,
    ) -> PoseEstimationPipeline:
        """Instantiate the prediction pipeline of this model.

        :param iou:     (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:    (Optional) Below the confidence threshold, prediction are discarded.
                        If None, the default value associated to the training is used.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        if None in (self._image_processor, self._default_nms_iou, self._default_nms_conf, self._edge_links):
            raise RuntimeError(
                "You must set the dataset processing parameters before calling predict.\n" "Please call `model.set_dataset_processing_params(...)` first."
            )

        iou = iou or self._default_nms_iou
        conf = conf or self._default_nms_conf
        pre_nms_max_predictions = pre_nms_max_predictions or self._default_pre_nms_max_predictions
        post_nms_max_predictions = post_nms_max_predictions or self._default_post_nms_max_predictions

        pipeline = PoseEstimationPipeline(
            model=self,
            image_processor=self._image_processor,
            post_prediction_callback=self.get_post_prediction_callback(
                iou=iou,
                conf=conf,
                pre_nms_max_predictions=pre_nms_max_predictions,
                post_nms_max_predictions=post_nms_max_predictions,
            ),
            fuse_model=fuse_model,
            edge_links=self._edge_links,
            edge_colors=self._edge_colors,
            keypoint_colors=self._keypoint_colors,
        )
        return pipeline

    @classmethod
    def get_post_prediction_callback(
        cls, conf: float, iou: float, pre_nms_max_predictions=1000, post_nms_max_predictions=300
    ) -> YoloNASPosePostPredictionCallback:
        return YoloNASPosePostPredictionCallback(
            pose_confidence_threshold=conf,
            nms_iou_threshold=iou,
            pre_nms_max_predictions=pre_nms_max_predictions,
            post_nms_max_predictions=post_nms_max_predictions,
        )

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
        iou: Optional[float] = 0.7,
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
        self._default_nms_iou = iou or self._default_nms_iou


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

    @property
    def num_classes(self):
        return self.heads.num_classes


@register_model(Models.YOLO_NAS_POSE_SHARED_S)
class YoloNASPoseShared_S(YoloNASPose):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("yolo_nas_pose_s_shared_arch_params")
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

    @property
    def num_classes(self):
        return self.heads.num_classes


@register_model(Models.YOLO_NAS_POSE_SHARED_M)
class YoloNASPoseShared_M(YoloNASPose):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("yolo_nas_pose_m_shared_arch_params")
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

    @property
    def num_classes(self):
        return self.heads.num_classes
