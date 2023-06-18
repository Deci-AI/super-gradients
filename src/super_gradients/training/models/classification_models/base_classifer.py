from functools import lru_cache

from super_gradients.training.models import SgModule
from super_gradients.training.pipelines.pipelines import ClassificationPipeline
from super_gradients.training.utils.media.image import ImageSource
from super_gradients.training.utils.predict import ImagesPredictions


class BaseClassifier(SgModule):
    def __init__(
        self,
    ):
        super(BaseClassifier, self).__init__()

    @lru_cache(maxsize=1)
    def _get_pipeline(self, fuse_model: bool = True) -> ClassificationPipeline:
        """Instantiate the prediction pipeline of this model.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        if None in (self._class_names, self._image_processor):
            raise RuntimeError(
                "You must set the dataset processing parameters before calling predict.\n" "Please call `model.set_dataset_processing_params(...)` first."
            )

        pipeline = ClassificationPipeline(
            model=self,
            image_processor=self._image_processor,
            class_names=self._class_names,
            fuse_model=fuse_model,
        )
        return pipeline

    def predict(self, images: ImageSource, fuse_model: bool = True) -> ImagesPredictions:
        """Predict an image or a list of images.

        :param images:  Images to predict.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        pipeline = self._get_pipeline(fuse_model=fuse_model)
        return pipeline(images)  # type: ignore

    def predict_webcam(self, fuse_model: bool = True):
        """Predict using webcam.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        pipeline = self._get_pipeline(fuse_model=fuse_model)
        pipeline.predict_webcam()
