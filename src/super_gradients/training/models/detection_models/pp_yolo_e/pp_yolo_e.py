from typing import Union, Optional, List
from functools import lru_cache

import torch
from torch import Tensor

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.processing_factory import ProcessingFactory
from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.modules import RepVGGBlock
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.models.detection_models.csp_resnet import CSPResNetBackbone
from super_gradients.training.models.detection_models.pp_yolo_e.pan import PPYoloECSPPAN
from super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_head import PPYOLOEHead
from super_gradients.training.utils import HpmStruct
from super_gradients.training.models.arch_params_factory import get_arch_params
from super_gradients.training.models.detection_models.pp_yolo_e.post_prediction_callback import PPYoloEPostPredictionCallback, DetectionPostPredictionCallback
from super_gradients.training.utils.predict import ImagesDetectionPrediction
from super_gradients.training.pipelines.pipelines import DetectionPipeline
from super_gradients.training.processing.processing import Processing
from super_gradients.training.utils.media.image import ImageSource


class PPYoloE(SgModule):
    def __init__(self, arch_params):
        super().__init__()
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()

        self.backbone = CSPResNetBackbone(**arch_params["backbone"], depth_mult=arch_params["depth_mult"], width_mult=arch_params["width_mult"])
        self.neck = PPYoloECSPPAN(**arch_params["neck"], depth_mult=arch_params["depth_mult"], width_mult=arch_params["width_mult"])
        self.head = PPYOLOEHead(**arch_params["head"], width_mult=arch_params["width_mult"], num_classes=arch_params["num_classes"])

        self._class_names: Optional[List[str]] = None
        self._image_processor: Optional[Processing] = None
        self._default_nms_iou: Optional[float] = None
        self._default_nms_conf: Optional[float] = None

    @staticmethod
    def get_post_prediction_callback(conf: float, iou: float) -> DetectionPostPredictionCallback:
        return PPYoloEPostPredictionCallback(score_threshold=conf, nms_threshold=iou, nms_top_k=1000, max_predictions=300)

    @resolve_param("image_processor", ProcessingFactory())
    def set_dataset_processing_params(
        self,
        class_names: Optional[List[str]] = None,
        image_processor: Optional[Processing] = None,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
    ) -> None:
        """Set the processing parameters for the dataset.

        :param class_names:     (Optional) Names of the dataset the model was trained on.
        :param image_processor: (Optional) Image processing objects to reproduce the dataset preprocessing used for training.
        :param iou:             (Optional) IoU threshold for the nms algorithm
        :param conf:            (Optional) Below the confidence threshold, prediction are discarded
        """
        self._class_names = class_names or self._class_names
        self._image_processor = image_processor or self._image_processor
        self._default_nms_iou = iou or self._default_nms_iou
        self._default_nms_conf = conf or self._default_nms_conf

    @lru_cache(maxsize=1)
    def _get_pipeline(self, iou: Optional[float] = None, conf: Optional[float] = None, fuse_model: bool = True) -> DetectionPipeline:
        """Instantiate the prediction pipeline of this model.

        :param iou:     (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:    (Optional) Below the confidence threshold, prediction are discarded.
                        If None, the default value associated to the training is used.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        if None in (self._class_names, self._image_processor, self._default_nms_iou, self._default_nms_conf):
            raise RuntimeError(
                "You must set the dataset processing parameters before calling predict.\n" "Please call `model.set_dataset_processing_params(...)` first."
            )

        iou = iou or self._default_nms_iou
        conf = conf or self._default_nms_conf

        pipeline = DetectionPipeline(
            model=self,
            image_processor=self._image_processor,
            post_prediction_callback=self.get_post_prediction_callback(iou=iou, conf=conf),
            class_names=self._class_names,
        )
        return pipeline

    def predict(self, images: ImageSource, iou: Optional[float] = None, conf: Optional[float] = None, fuse_model: bool = True) -> ImagesDetectionPrediction:
        """Predict an image or a list of images.

        :param images:  Images to predict.
        :param iou:     (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:    (Optional) Below the confidence threshold, prediction are discarded.
                        If None, the default value associated to the training is used.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        pipeline = self._get_pipeline(iou=iou, conf=conf, fuse_model=fuse_model)
        return pipeline(images)  # type: ignore

    def predict_webcam(self, iou: Optional[float] = None, conf: Optional[float] = None, fuse_model: bool = True):
        """Predict using webcam.

        :param iou:     (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:    (Optional) Below the confidence threshold, prediction are discarded.
                        If None, the default value associated to the training is used.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        pipeline = self._get_pipeline(iou=iou, conf=conf, fuse_model=fuse_model)
        pipeline.predict_webcam()

    def train(self, mode: bool = True):
        self._get_pipeline.cache_clear()
        torch.cuda.empty_cache()
        return super().train(mode)

    def forward(self, x: Tensor):
        features = self.backbone(x)
        features = self.neck(features)
        return self.head(features)

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        """
        Prepare the model to be converted to ONNX or other frameworks.
        Typically, this function will freeze the size of layers which is otherwise flexible, replace some modules
        with convertible substitutes and remove all auxiliary or training related parts.
        :param input_size: [H,W]
        """
        self.head.cache_anchors(input_size)

        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.fuse_block_residual_branches()

    def replace_head(self, new_num_classes=None, new_head=None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_head is not None:
            self.head = new_head
        else:
            self.head.replace_num_classes(new_num_classes)


@register_model(Models.PP_YOLOE_S)
class PPYoloE_S(PPYoloE):
    def __init__(self, arch_params):
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()
        arch_params = get_arch_params("ppyoloe_s_arch_params", overriding_params=arch_params)
        super().__init__(arch_params)


@register_model(Models.PP_YOLOE_M)
class PPYoloE_M(PPYoloE):
    def __init__(self, arch_params):
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()
        arch_params = get_arch_params("ppyoloe_m_arch_params", overriding_params=arch_params)
        super().__init__(arch_params)


@register_model(Models.PP_YOLOE_L)
class PPYoloE_L(PPYoloE):
    def __init__(self, arch_params):
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()
        arch_params = get_arch_params("ppyoloe_l_arch_params", overriding_params=arch_params)
        super().__init__(arch_params)


@register_model(Models.PP_YOLOE_X)
class PPYoloE_X(PPYoloE):
    def __init__(self, arch_params):
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()
        arch_params = get_arch_params("ppyoloe_x_arch_params", overriding_params=arch_params)
        super().__init__(arch_params)
