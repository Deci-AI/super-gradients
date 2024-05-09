import copy
from functools import lru_cache
from typing import Union, Optional, Tuple, Any

import torch
from omegaconf import DictConfig
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.processing_factory import ProcessingFactory
from super_gradients.common.object_names import Models
from super_gradients.common.registry import register_model
from super_gradients.module_interfaces import AbstractPoseEstimationDecodingModule, SupportsInputShapeCheck
from super_gradients.training.models.arch_params_factory import get_arch_params
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.models.detection_models.yolo_nas_r.yolo_nas_r_post_prediction_callback import YoloNASRPostPredictionCallback
from super_gradients.training.pipelines import OBBDetectionPipeline
from super_gradients.training.processing import Processing, ComposeProcessing, OBBDetectionAutoPadding
from super_gradients.training.utils import get_param
from super_gradients.training.utils.media.image import ImageSource
from super_gradients.training.utils.utils import HpmStruct
from torch import Tensor

logger = get_logger(__name__)


class YoloNASRDecodingModule(AbstractPoseEstimationDecodingModule):
    __constants__ = ["num_pre_nms_predictions"]

    def __init__(
        self,
        num_pre_nms_predictions: int = 1000,
    ):
        super().__init__()
        self.num_pre_nms_predictions = num_pre_nms_predictions

    @torch.jit.ignore
    def infer_total_number_of_predictions(self, inputs: Any) -> int:
        """

        :param inputs: YoloNASPose model outputs
        :return:
        """
        if torch.jit.is_tracing():
            pred_bboxes_xyxy, pred_bboxes_conf, pred_pose_coords, pred_pose_scores = inputs
        else:
            pred_bboxes_xyxy, pred_bboxes_conf, pred_pose_coords, pred_pose_scores = inputs[0]

        return pred_bboxes_xyxy.size(1)

    def get_num_pre_nms_predictions(self) -> int:
        return self.num_pre_nms_predictions

    def forward(self, inputs: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, ...]]):
        """
        Decode YoloNASPose model outputs into bounding boxes, confidence scores and pose coordinates and scores

        :param inputs: YoloNASPose model outputs
        :return: Tuple of (pred_bboxes, pred_scores, pred_joints)
        - pred_bboxes: [Batch, num_pre_nms_predictions, 4] Bounding of associated with pose in XYXY format
        - pred_scores: [Batch, num_pre_nms_predictions, 1] Confidence scores [0..1] for entire pose
        - pred_joints: [Batch, num_pre_nms_predictions, Num Joints, 3] Joints in (x,y,confidence) format
        """
        if torch.jit.is_tracing():
            pred_bboxes_xyxy, pred_bboxes_conf, pred_pose_coords, pred_pose_scores = inputs
        else:
            pred_bboxes_xyxy, pred_bboxes_conf, pred_pose_coords, pred_pose_scores = inputs[0]

        nms_top_k = self.num_pre_nms_predictions
        batch_size, num_anchors, _ = pred_bboxes_conf.size()

        topk_candidates = torch.topk(pred_bboxes_conf, dim=1, k=nms_top_k, largest=True, sorted=True)

        offsets = num_anchors * torch.arange(batch_size, device=pred_bboxes_conf.device)
        indices_with_offset = topk_candidates.indices + offsets.reshape(batch_size, 1, 1)
        flat_indices = torch.flatten(indices_with_offset)

        pred_poses_and_scores = torch.cat([pred_pose_coords, pred_pose_scores.unsqueeze(3)], dim=3)

        output_pred_bboxes = pred_bboxes_xyxy.reshape(-1, pred_bboxes_xyxy.size(2))[flat_indices, :].reshape(
            pred_bboxes_xyxy.size(0), nms_top_k, pred_bboxes_xyxy.size(2)
        )
        output_pred_scores = pred_bboxes_conf.reshape(-1, pred_bboxes_conf.size(2))[flat_indices, :].reshape(
            pred_bboxes_conf.size(0), nms_top_k, pred_bboxes_conf.size(2)
        )
        output_pred_joints = pred_poses_and_scores.reshape(-1, pred_poses_and_scores.size(2), 3)[flat_indices, :, :].reshape(
            pred_poses_and_scores.size(0), nms_top_k, pred_poses_and_scores.size(2), pred_poses_and_scores.size(3)
        )

        return output_pred_bboxes, output_pred_scores, output_pred_joints


class YoloNASR(CustomizableDetector, SupportsInputShapeCheck):
    """
    YoloNASR model
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
        super().__init__(
            backbone=backbone,
            heads=heads,
            neck=neck,
            num_classes=num_classes,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
            inplace_act=inplace_act,
            in_channels=in_channels,
        )
        self._image_processor = None
        self._default_nms_conf = None
        self._default_nms_iou = None
        self._default_pre_nms_max_predictions = None
        self._default_post_nms_max_predictions = None

    def get_decoding_module(self, num_pre_nms_predictions: int, **kwargs) -> AbstractPoseEstimationDecodingModule:
        return YoloNASRDecodingModule(num_pre_nms_predictions)

    def predict(
        self,
        images: ImageSource,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        pre_nms_max_predictions: Optional[int] = None,
        post_nms_max_predictions: Optional[int] = None,
        batch_size: int = 32,
        fuse_model: bool = True,
        skip_image_resizing: bool = False,
        fp16: bool = True,
    ):
        """Predict an image or a list of images.

        :param images:     Images to predict.
        :param iou:        (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:       (Optional) Below the confidence threshold, prediction are discarded.
                           If None, the default value associated to the training is used.
        :param batch_size: Maximum number of images to process at the same time.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        :param fp16:       If True, use mixed precision for inference.
        """
        pipeline = self._get_pipeline(
            iou=iou,
            conf=conf,
            pre_nms_max_predictions=pre_nms_max_predictions,
            post_nms_max_predictions=post_nms_max_predictions,
            fuse_model=fuse_model,
            skip_image_resizing=skip_image_resizing,
            fp16=fp16,
        )
        return pipeline(images, batch_size=batch_size)  # type: ignore

    def predict_webcam(
        self,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        pre_nms_max_predictions: Optional[int] = None,
        post_nms_max_predictions: Optional[int] = None,
        batch_size: int = 32,
        fuse_model: bool = True,
        skip_image_resizing: bool = False,
        fp16: bool = True,
    ):
        """Predict using webcam.

        :param iou:        (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:       (Optional) Below the confidence threshold, prediction are discarded.
                           If None, the default value associated to the training is used.
        :param batch_size: Maximum number of images to process at the same time.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        :param fp16:       If True, use mixed precision for inference.

        """
        pipeline = self._get_pipeline(
            iou=iou,
            conf=conf,
            pre_nms_max_predictions=pre_nms_max_predictions,
            post_nms_max_predictions=post_nms_max_predictions,
            fuse_model=fuse_model,
            skip_image_resizing=skip_image_resizing,
            fp16=fp16,
        )
        pipeline.predict_webcam()

    @lru_cache(1)
    def _get_pipeline(
        self,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        pre_nms_max_predictions: Optional[int] = None,
        post_nms_max_predictions: Optional[int] = None,
        fuse_model: bool = True,
        skip_image_resizing: bool = False,
        fp16: bool = True,
    ) -> OBBDetectionPipeline:
        """Instantiate the prediction pipeline of this model.

        :param iou:        (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:       (Optional) Below the confidence threshold, prediction are discarded.
                           If None, the default value associated to the training is used.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        :param fp16:       If True, use mixed precision for inference.
        """
        if None in (self._image_processor, self._class_names, self._default_nms_iou, self._default_nms_conf):
            raise RuntimeError(
                "You must set the dataset processing parameters before calling predict.\n" "Please call `model.set_dataset_processing_params(...)` first."
            )

        iou = iou or self._default_nms_iou
        conf = conf or self._default_nms_conf
        pre_nms_max_predictions = pre_nms_max_predictions or self._default_pre_nms_max_predictions
        post_nms_max_predictions = post_nms_max_predictions or self._default_post_nms_max_predictions

        # Ensure that the image size is divisible by 32.
        if isinstance(self._image_processor, ComposeProcessing) and skip_image_resizing:
            image_processor = self._image_processor.get_equivalent_compose_without_resizing(
                auto_padding=OBBDetectionAutoPadding(shape_multiple=(32, 32), pad_value=0)
            )
        else:
            image_processor = self._image_processor

        pipeline = OBBDetectionPipeline(
            model=self,
            class_names=self._class_names,
            image_processor=image_processor,
            post_prediction_callback=self.get_post_prediction_callback(
                iou=iou,
                conf=conf,
                pre_nms_max_predictions=pre_nms_max_predictions,
                post_nms_max_predictions=post_nms_max_predictions,
            ),
            fuse_model=fuse_model,
            fp16=fp16,
        )
        return pipeline

    @classmethod
    def get_post_prediction_callback(
        cls, conf: float, iou: float, pre_nms_max_predictions=1000, post_nms_max_predictions=300
    ) -> YoloNASRPostPredictionCallback:
        return YoloNASRPostPredictionCallback(
            score_threshold=conf,
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
        image_processor: Optional[Processing] = None,
        class_names=None,
        conf: Optional[float] = None,
        iou: Optional[float] = 0.7,
        pre_nms_max_predictions=300,
        post_nms_max_predictions=100,
    ) -> None:
        """Set the processing parameters for the dataset.

        :param image_processor: (Optional) Image processing objects to reproduce the dataset preprocessing used for training.
        :param conf:            (Optional) Below the confidence threshold, prediction are discarded
        """
        self._image_processor = image_processor or self._image_processor
        self._class_names = list(class_names) if class_names is not None else self._class_names
        self._default_nms_conf = conf or self._default_nms_conf
        self._default_nms_iou = iou or self._default_nms_iou
        self._default_pre_nms_max_predictions = pre_nms_max_predictions or self._default_pre_nms_max_predictions
        self._default_post_nms_max_predictions = post_nms_max_predictions or self._default_post_nms_max_predictions

    def get_input_shape_steps(self) -> Tuple[int, int]:
        """
        Returns the minimum input shape size that the model can accept.
        For segmentation models the default is 32x32, which corresponds to the largest stride in the encoder part of the model
        """
        return 32, 32

    def get_minimum_input_shape_size(self) -> Tuple[int, int]:
        """
        Returns the minimum input shape size that the model can accept.
        For segmentation models the default is 32x32, which corresponds to the largest stride in the encoder part of the model
        """
        return 32, 32


@register_model(Models.YOLO_NAS_R_S)
class YoloNASR_S(YoloNASR):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("yolo_nas_r_s_arch_params")
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


@register_model(Models.YOLO_NAS_R_M)
class YoloNASR_M(YoloNASR):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("yolo_nas_r_m_arch_params")
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


@register_model(Models.YOLO_NAS_R_L)
class YoloNASR_L(YoloNASR):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("yolo_nas_r_l_arch_params")
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
