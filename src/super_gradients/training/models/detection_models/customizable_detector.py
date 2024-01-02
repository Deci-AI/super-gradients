"""
A base for a detection network built according to the following scheme:
 * constructed from nested arch_params;
 * inside arch_params each nested level (module) has an explicit type and its required parameters
 * each module accepts in_channels and other parameters
 * each module defines out_channels property on construction
"""
from typing import Union, Optional, List, Callable
from functools import lru_cache

import torch
from torch import nn
from omegaconf import DictConfig

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.processing_factory import ProcessingFactory
from super_gradients.module_interfaces import SupportsReplaceInputChannels, HasPredict
from super_gradients.modules import ConvBNAct
from super_gradients.training.utils.utils import HpmStruct, arch_params_deprecated
from super_gradients.training.models.sg_module import SgModule
import super_gradients.common.factories.detection_modules_factory as det_factory
from super_gradients.training.utils.predict import ImagesDetectionPrediction
from super_gradients.training.pipelines.pipelines import DetectionPipeline
from super_gradients.training.processing.processing import Processing, ComposeProcessing, DetectionAutoPadding
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback
from super_gradients.training.utils.media.image import ImageSource


class CustomizableDetector(HasPredict, SgModule):
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
        num_branches: int = 5,
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
        :param num_branches: Number of branches.
        :param num_classes: num classes to predict.
        :param bn_eps:      Epsilon for batch norm.
        :param bn_momentum: Momentum for batch norm.
        :param inplace_act: If True, do the operations operation in-place when possible.
        :param in_channels: number of input channels
        """
        super().__init__()

        self.phase = 1
        self.heads_params = heads
        self.neck_params = neck
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.inplace_act = inplace_act
        self.in_channels = in_channels
        self.num_branches = num_branches
        factory = det_factory.DetectionModulesFactory()

        # move num_classes into heads params
        if num_classes is not None:
            self.heads_params = factory.insert_module_param(self.heads_params, "num_classes", num_classes)

        self.backbone = factory.get(factory.insert_module_param(backbone, "in_channels", in_channels))
        self.necks = nn.ModuleList()
        self.heads_sets = nn.ModuleList()
        if neck is not None:
            for i in range(num_branches):
                neck = factory.get(factory.insert_module_param(self.neck_params, "in_channels", self.backbone.out_channels))
                heads = factory.get(factory.insert_module_param(self.heads_params, "in_channels", neck.out_channels))
                self.necks.append(neck)
                self.heads_sets.append(heads)
        else:
            for i in range(num_branches):
                neck = nn.Identity()
                heads = factory.get(factory.insert_module_param(heads, "in_channels", self.backbone.out_channels))
                self.necks.append(neck)
                self.heads_sets.append(heads)

        self.selector = self._get_selector()

        self._initialize_weights(bn_eps, bn_momentum, inplace_act)

        # Processing params
        self._class_names: Optional[List[str]] = None
        self._image_processor: Optional[Processing] = None
        self._default_nms_iou: Optional[float] = None
        self._default_nms_conf: Optional[float] = None

    def _get_selector(self):
        return nn.Sequential(
            ConvBNAct(self.backbone.out_channels[-1], self.backbone.out_channels[-1], 1, 0, activation_type=nn.ReLU),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels[-1], self.num_branches),
        )

    def set_phase(self, phase: int):
        self.phase = phase

    def forward(self, x):
        x = self.backbone(x)
        selection = self.selector(x[-1])

        if self.phase == 2 or not self.training:
            selection = torch.argmax(selection, dim=1)

            xs = self._split_batch(x)
            list_of_outputs = []

            for _x, sel in zip(xs, selection):
                _x = self.necks[sel](_x)
                _x = self.heads_sets[sel](_x)
                list_of_outputs.append(_x)

            if not self.training:  # if in eval mode
                outputs = (
                    self._merge_batch([list_of_outputs[i][0] for i in range(len(list_of_outputs))]),
                    selection,
                    self._merge_batch([list_of_outputs[i][1] for i in range(len(list_of_outputs))]),
                )
            else:
                outputs = selection, self._merge_batch(list_of_outputs)
            return outputs
        elif self.phase == 1:
            xs = []
            for neck in self.necks:
                xs.append(neck(x))

            outputs = []
            for head, x in zip(self.heads_sets, xs):
                outputs.append(head(x))

            return selection, outputs
        else:
            raise ValueError(f"unsupported phase: {self.phase}")

    @staticmethod
    def _split_batch(outputs: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        list_of_outputs = []
        for index_in_batch in range(len(outputs[0])):
            sliced_outputs = []
            for output_index in range(len(outputs)):
                sliced_outputs.append(outputs[output_index][index_in_batch].unsqueeze(0))
            list_of_outputs.append(sliced_outputs)

        return list_of_outputs

    @staticmethod
    def _merge_batch(list_of_outputs: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        outputs = []
        for output_index in range(len(list_of_outputs[0])):
            if isinstance(list_of_outputs[0][output_index], list) or list_of_outputs[0][output_index].dim() < 3:
                outputs.append(list_of_outputs[0][output_index])
                continue

            batch_items = [list_of_outputs[i][output_index] for i in range(len(list_of_outputs))]

            outputs.append(torch.cat(batch_items, dim=0))

        return outputs

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
        raise NotImplementedError("unsupported method: replace_head")

    def replace_input_channels(self, in_channels: int, compute_new_weights_fn: Optional[Callable[[nn.Module, int], nn.Module]] = None):
        if isinstance(self.backbone, SupportsReplaceInputChannels):
            self.backbone.replace_input_channels(in_channels=in_channels, compute_new_weights_fn=compute_new_weights_fn)
            self.in_channels = self.get_input_channels()
        else:
            raise NotImplementedError(f"`{self.backbone.__class__.__name__}` does not support `replace_input_channels`")

    def get_input_channels(self) -> int:
        if isinstance(self.backbone, SupportsReplaceInputChannels):
            return self.backbone.get_input_channels()
        else:
            raise NotImplementedError(f"`{self.backbone.__class__.__name__}` does not support `replace_input_channels`")

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

    def get_processing_params(self) -> Optional[Processing]:
        return self._image_processor

    @lru_cache(maxsize=1)
    def _get_pipeline(
        self, iou: Optional[float] = None, conf: Optional[float] = None, fuse_model: bool = True, skip_image_resizing: bool = False
    ) -> DetectionPipeline:
        """Instantiate the prediction pipeline of this model.

        :param iou:     (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:    (Optional) Below the confidence threshold, prediction are discarded.
                        If None, the default value associated to the training is used.
        :param fuse_model:  If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        """
        if None in (self._class_names, self._image_processor, self._default_nms_iou, self._default_nms_conf):
            raise RuntimeError(
                "You must set the dataset processing parameters before calling predict.\n" "Please call `model.set_dataset_processing_params(...)` first."
            )

        iou = iou or self._default_nms_iou
        conf = conf or self._default_nms_conf

        # Ensure that the image size is divisible by 32.
        if isinstance(self._image_processor, ComposeProcessing) and skip_image_resizing:
            image_processor = self._image_processor.get_equivalent_compose_without_resizing(
                auto_padding=DetectionAutoPadding(shape_multiple=(32, 32), pad_value=0)
            )
        else:
            image_processor = self._image_processor

        pipeline = DetectionPipeline(
            model=self,
            image_processor=image_processor,
            post_prediction_callback=self.get_post_prediction_callback(iou=iou, conf=conf),
            class_names=self._class_names,
            fuse_model=fuse_model,
        )
        return pipeline

    def predict(
        self,
        images: ImageSource,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        batch_size: int = 32,
        fuse_model: bool = True,
        skip_image_resizing: bool = False,
    ) -> ImagesDetectionPrediction:
        """Predict an image or a list of images.

        :param images:      Images to predict.
        :param iou:         (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:        (Optional) Below the confidence threshold, prediction are discarded.
                            If None, the default value associated to the training is used.
        :param batch_size:  Maximum number of images to process at the same time.
        :param fuse_model:  If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        """
        pipeline = self._get_pipeline(iou=iou, conf=conf, fuse_model=fuse_model, skip_image_resizing=skip_image_resizing)
        return pipeline(images, batch_size=batch_size)  # type: ignore

    def predict_webcam(self, iou: Optional[float] = None, conf: Optional[float] = None, fuse_model: bool = True, skip_image_resizing: bool = False):
        """Predict using webcam.

        :param iou:     (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:    (Optional) Below the confidence threshold, prediction are discarded.
                        If None, the default value associated to the training is used.
        :param fuse_model:  If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        """
        pipeline = self._get_pipeline(iou=iou, conf=conf, fuse_model=fuse_model, skip_image_resizing=skip_image_resizing)
        pipeline.predict_webcam()

    def train(self, mode: bool = True):
        self._get_pipeline.cache_clear()
        torch.cuda.empty_cache()
        return super().train(mode)

    def get_finetune_lr_dict(self, lr: float):
        return {"heads": lr, "default": 0}
