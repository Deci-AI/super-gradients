from typing import Optional, List
from functools import lru_cache

from super_gradients.module_interfaces import HasPredict
from super_gradients.training.models import SgModule
from super_gradients.training.pipelines.pipelines import ClassificationPipeline
from super_gradients.training.utils.media.image import ImageSource
from super_gradients.training.utils.predict import ImagesClassificationPrediction
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.processing_factory import ProcessingFactory
from super_gradients.training.processing.processing import Processing


class BaseClassifier(SgModule, HasPredict):
    def __init__(
        self,
    ):
        self._class_names: Optional[List[str]] = None
        self._image_processor: Optional[Processing] = None
        super(BaseClassifier, self).__init__()

    @resolve_param("image_processor", ProcessingFactory())
    def set_dataset_processing_params(self, class_names: Optional[List[str]] = None, image_processor: Optional[Processing] = None) -> None:
        """Set the processing parameters for the dataset.

        :param class_names:     (Optional) Names of the dataset the model was trained on.
        :param image_processor: (Optional) Image processing objects to reproduce the dataset preprocessing used for training.
        """
        self._class_names = class_names or self._class_names
        self._image_processor = image_processor or self._image_processor

    @lru_cache(maxsize=1)
    def _get_pipeline(self, fuse_model: bool = True, skip_image_resizing: bool = False, fp16: bool = True) -> ClassificationPipeline:
        """Instantiate the prediction pipeline of this model.
        :param fuse_model:  If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        :param fp16: If True, use mixed precision for inference.
        """
        if None in (self._class_names, self._image_processor):
            raise RuntimeError(
                "You must set the dataset processing parameters before calling predict.\n" "Please call `model.set_dataset_processing_params(...)` first."
            )

        if skip_image_resizing:
            raise ValueError("`skip_image_resizing` is not supported for classification models.")

        pipeline = ClassificationPipeline(
            model=self,
            image_processor=self._image_processor,
            class_names=self._class_names,
            fuse_model=fuse_model,
            fp16=fp16,
        )
        return pipeline

    def predict(
        self,
        images: ImageSource,
        batch_size: int = 32,
        fuse_model: bool = True,
        skip_image_resizing: bool = False,
        fp16: bool = True,
    ) -> ImagesClassificationPrediction:
        """Predict an image or a list of images.

        :param images:      Images to predict.
        :param batch_size:  Maximum number of images to process at the same time.
        :param fuse_model:  If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        :param fp16: If True, use mixed precision for inference.
        """
        pipeline = self._get_pipeline(fuse_model=fuse_model, skip_image_resizing=skip_image_resizing, fp16=fp16)
        return pipeline(images, batch_size=batch_size)  # type: ignore

    def predict_webcam(self, fuse_model: bool = True, skip_image_resizing: bool = False, fp16: bool = True) -> None:
        """Predict using webcam.
        :param fuse_model:  If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        :param fp16: If True, use mixed precision for inference.
        """
        pipeline = self._get_pipeline(fuse_model=fuse_model, skip_image_resizing=skip_image_resizing, fp16=fp16)
        pipeline.predict_webcam()
