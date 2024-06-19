from typing import List, Optional, Union, Iterable

import numpy as np
import torch
from super_gradients.module_interfaces import AbstractOBBPostPredictionCallback, OBBPredictions
from super_gradients.training.models import SgModule
from super_gradients.training.processing import ComposeProcessing
from super_gradients.training.processing.processing import Processing, ImagePermute
from super_gradients.training.utils.predict import (
    OBBDetectionPrediction,
    ImageOBBDetectionPrediction,
    ImagesOBBDetectionPrediction,
    VideoOBBDetectionPrediction,
)
from tqdm import tqdm

from .pipelines import Pipeline

__all__ = ["OBBDetectionPipeline"]


class OBBDetectionPipeline(Pipeline):
    """
    Pipeline specifically designed for oriented object detection task.
    The pipeline includes loading images, preprocessing, prediction, and postprocessing.

    :param model:                       The object detection model (instance of SgModule) used for making predictions.
    :param class_names:                 List of class names corresponding to the model's output classes.
    :param post_prediction_callback:    Callback function to process raw predictions from the model.
    :param image_processor:             Single image processor or a list of image processors for preprocessing and postprocessing the images.
    :param device:                      The device on which the model will be run. If None, will run on current model device. Use "cuda" for GPU support.
    :param fuse_model:                  If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
    :param fp16:                        If True, use mixed precision for inference.
    """

    def __init__(
        self,
        model: SgModule,
        class_names: List[str],
        post_prediction_callback: AbstractOBBPostPredictionCallback,
        device: Optional[str] = None,
        image_processor: Union[Processing, List[Processing]] = None,
        fuse_model: bool = True,
        fp16: bool = True,
    ):
        if isinstance(image_processor, list):
            image_processor = ComposeProcessing(image_processor)

        has_image_permute = any(isinstance(image_processing, ImagePermute) for image_processing in image_processor.processings)
        if not has_image_permute:
            image_processor.processings.append(ImagePermute())

        super().__init__(
            model=model,
            device=device,
            image_processor=image_processor,
            class_names=class_names,
            fuse_model=fuse_model,
            fp16=fp16,
        )
        self.post_prediction_callback = post_prediction_callback

    def _decode_model_output(self, model_output, model_input: np.ndarray) -> List[OBBDetectionPrediction]:
        """Decode the model output, by applying post prediction callback. This includes NMS.

        :param model_output:    Direct output of the model, without any post-processing.
        :param model_input:     Model input (i.e. images after preprocessing).
        :return:                Predicted Bboxes.
        """
        post_nms_predictions: List[OBBPredictions] = self.post_prediction_callback(model_output)

        predictions = []
        for prediction, image in zip(post_nms_predictions, model_input):
            predictions.append(
                OBBDetectionPrediction(
                    rboxes_cxcywhr=(
                        prediction.rboxes_cxcywhr.detach().cpu().numpy() if torch.is_tensor(prediction.rboxes_cxcywhr) else prediction.rboxes_cxcywhr
                    ),
                    confidence=prediction.scores.detach().cpu().numpy() if torch.is_tensor(prediction.scores) else prediction.scores,
                    labels=prediction.labels.detach().cpu().numpy() if torch.is_tensor(prediction.labels) else prediction.labels,
                    image_shape=image.shape,
                )
            )

        return predictions

    def _instantiate_image_prediction(self, image: np.ndarray, prediction: OBBDetectionPrediction) -> ImageOBBDetectionPrediction:
        return ImageOBBDetectionPrediction(image=image, prediction=prediction, class_names=self.class_names)

    def _combine_image_prediction_to_images(
        self, images_predictions: Iterable[ImageOBBDetectionPrediction], n_images: Optional[int] = None
    ) -> Union[ImagesOBBDetectionPrediction, ImageOBBDetectionPrediction]:
        if n_images is not None and n_images == 1:
            # Do not show tqdm progress bar if there is only one image
            images_predictions = next(iter(images_predictions))
        else:
            images_predictions = [image_predictions for image_predictions in tqdm(images_predictions, total=n_images, desc="Predicting Images")]
            images_predictions = ImagesOBBDetectionPrediction(_images_prediction_lst=images_predictions)

        return images_predictions

    def _combine_image_prediction_to_video(
        self, images_predictions: Iterable[ImageOBBDetectionPrediction], fps: float, n_images: Optional[int] = None
    ) -> VideoOBBDetectionPrediction:
        return VideoOBBDetectionPrediction(_images_prediction_gen=images_predictions, fps=fps, n_frames=n_images)
