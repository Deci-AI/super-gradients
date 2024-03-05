from super_gradients.module_interfaces import SupportsInputShapeCheck
from super_gradients.training.models.sg_module import SgModule
import torch.nn as nn
from abc import abstractmethod, ABC
from typing import Union, List, Optional, Tuple
from functools import lru_cache
from super_gradients.training.pipelines.pipelines import SegmentationPipeline
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.processing_factory import ProcessingFactory
from super_gradients.training.processing.processing import Processing
from super_gradients.training.utils.predict import ImagesSegmentationPrediction
from super_gradients.training.utils.media.image import ImageSource
from super_gradients.module_interfaces import HasPredict


class SegmentationModule(SgModule, ABC, HasPredict, SupportsInputShapeCheck):
    """
    Base SegmentationModule class
    """

    def __init__(self, use_aux_heads: bool):
        super().__init__()
        self._use_aux_heads = use_aux_heads

        # Processing params
        self._class_names: Optional[List[str]] = None
        self._image_processor: Optional[Processing] = None

    @property
    def use_aux_heads(self):
        return self._use_aux_heads

    @use_aux_heads.setter
    def use_aux_heads(self, use_aux: bool):
        """
        public setter for self._use_aux_heads, called every time an assignment to self.use_aux_heads is applied.
        if use_aux is False, `_remove_auxiliary_heads` is called to delete auxiliary and detail heads.
        if use_aux is True, and self._use_aux_heads was already set to False a ValueError is raised, recreating
            aux and detail heads outside init method is not allowed, and the module should be recreated.
        """
        if use_aux is True and self._use_aux_heads is False:
            raise ValueError(
                "Cant turn use_aux_heads from False to True. Try initiating the module again with"
                " `use_aux_heads=True` or initiating the auxiliary heads modules manually."
            )
        if not use_aux:
            self._remove_auxiliary_heads()
        self._use_aux_heads = use_aux

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        # set to false and delete auxiliary and detail heads modules.
        self.use_aux_heads = False

    @abstractmethod
    def _remove_auxiliary_heads(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def backbone(self) -> nn.Module:
        """
        For SgTrainer load_backbone compatibility.
        """
        raise NotImplementedError()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @resolve_param("image_processor", ProcessingFactory())
    def set_dataset_processing_params(
        self,
        class_names: Optional[List[str]] = None,
        image_processor: Optional[Processing] = None,
    ) -> None:
        """Set the processing parameters for the dataset.

        :param class_names:     (Optional) Names of the dataset the model was trained on.
        :param image_processor: (Optional) Image processing objects to reproduce the dataset preprocessing used for training.
        """
        self._class_names = class_names or self._class_names
        self._image_processor = image_processor or self._image_processor

    @lru_cache(maxsize=1)
    def _get_pipeline(self, fuse_model: bool = True, fp16: bool = True) -> SegmentationPipeline:
        """Instantiate the segmentation pipeline of this model.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        if None in (self._class_names, self._image_processor):
            raise RuntimeError(
                "You must set the dataset processing parameters before calling predict.\n" "Please call `model.set_dataset_processing_params(...)` first."
            )

        pipeline = SegmentationPipeline(
            model=self,
            image_processor=self._image_processor,
            class_names=self._class_names,
            fuse_model=fuse_model,
            fp16=fp16,
        )
        return pipeline

    def predict(self, images: ImageSource, batch_size: int = 32, fuse_model: bool = True, fp16: bool = True) -> ImagesSegmentationPrediction:
        """Predict an image or a list of images.
        :param images:  Images to predict.
        :param batch_size:  Maximum number of images to process at the same time.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param fp16:                        If True, use mixed precision for inference.
        """
        pipeline = self._get_pipeline(fuse_model=fuse_model, fp16=fp16)
        return pipeline(images, batch_size=batch_size)  # type: ignore

    def predict_webcam(self, fuse_model: bool = True, fp16: bool = True):
        """Predict using webcam.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param fp16:       If True, use mixed precision for inference.
        """
        pipeline = self._get_pipeline(fuse_model=fuse_model, fp16=fp16)
        pipeline.predict_webcam()

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
