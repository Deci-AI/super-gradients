"""
A base for a detection network built according to the following scheme:
 * constructed from nested arch_params;
 * inside arch_params each nested level (module) has an explicit type and its required parameters
 * each module accepts in_channels and other parameters
 * each module defines out_channels property on construction
"""
from typing import Union, Optional, List
from functools import lru_cache

import torch
from torch import nn
from omegaconf import DictConfig

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.processing_factory import ProcessingFactory
from super_gradients.module_interfaces import SupportsReplaceNumClasses
from super_gradients.modules.head_replacement_utils import replace_num_classes_with_random_weights
from super_gradients.training.utils.utils import HpmStruct, arch_params_deprecated
from super_gradients.training.models.sg_module import SgModule
import super_gradients.common.factories.detection_modules_factory as det_factory
from super_gradients.training.utils.predict import ImagesDetectionPrediction
from super_gradients.training.pipelines.pipelines import DetectionPipeline
from super_gradients.training.processing.processing import Processing
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback
from super_gradients.training.utils.media.image import ImageSource


class CustomizableDetector(SgModule):
    """
    A customizable detector with backbone -> neck -> heads
    Each submodule with its parameters must be defined explicitly.
    Modules should follow the interface of BaseDetectionModule
    """

    @arch_params_deprecated
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
        """
        :param backbone:    Backbone configuration.
        :param heads:       Head configuration.
        :param neck:        Neck configuration.
        :param num_classes: num classes to predict.
        :param bn_eps:      Epsilon for batch norm.
        :param bn_momentum: Momentum for batch norm.
        :param inplace_act: If True, do the operations operation in-place when possible.
        :param in_channels: number of input channels
        """
        super().__init__()

        self.heads_params = heads
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.inplace_act = inplace_act
        factory = det_factory.DetectionModulesFactory()

        # move num_classes into heads params
        if num_classes is not None:
            self.heads_params = factory.insert_module_param(self.heads_params, "num_classes", num_classes)

        self.backbone = factory.get(factory.insert_module_param(backbone, "in_channels", in_channels))
        if neck is not None:
            self.neck = factory.get(factory.insert_module_param(neck, "in_channels", self.backbone.out_channels))
            self.heads = factory.get(factory.insert_module_param(heads, "in_channels", self.neck.out_channels))
        else:
            self.neck = nn.Identity()
            self.heads = factory.get(factory.insert_module_param(heads, "in_channels", self.backbone.out_channels))

        self._initialize_weights(bn_eps, bn_momentum, inplace_act)

        # Processing params
        self._class_names: Optional[List[str]] = None
        self._image_processor: Optional[Processing] = None
        self._default_nms_iou: Optional[float] = None
        self._default_nms_conf: Optional[float] = None

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.heads(x)

    def _initialize_weights(self, bn_eps: Optional[float] = None, bn_momentum: Optional[float] = None, inplace_act: Optional[bool] = True):
        for m in self.modules():
            t = type(m)
            if t is nn.BatchNorm2d:
                m.eps = bn_eps if bn_eps else m.eps
                m.momentum = bn_momentum if bn_momentum else m.momentum
            elif inplace_act and t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, nn.Mish]:
                m.inplace = True

    def prep_model_for_conversion(self, input_size: Optional[Union[tuple, list]] = None, **kwargs):
        for module in self.modules():
            if module != self and hasattr(module, "prep_model_for_conversion"):
                module.prep_model_for_conversion(input_size, **kwargs)

    def replace_head(self, new_num_classes: Optional[int] = None, new_head: Optional[nn.Module] = None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_head is not None:
            self.heads = new_head
        elif isinstance(self.heads, SupportsReplaceNumClasses):
            self.heads.replace_num_classes(new_num_classes, replace_num_classes_with_random_weights)
        else:
            factory = det_factory.DetectionModulesFactory()
            self.heads_params = factory.insert_module_param(self.heads_params, "num_classes", new_num_classes)
            self.heads = factory.get(factory.insert_module_param(self.heads_params, "in_channels", self.neck.out_channels))
            self._initialize_weights(self.bn_eps, self.bn_momentum, self.inplace_act)

    @staticmethod
    def get_post_prediction_callback(conf: float, iou: float) -> DetectionPostPredictionCallback:
        raise NotImplementedError

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
            fuse_model=fuse_model,
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
