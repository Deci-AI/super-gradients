import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Iterable
from contextlib import contextmanager
from tqdm import tqdm

import numpy as np
import torch

from super_gradients.training.utils.utils import generate_batch
from super_gradients.training.utils.media.videos import load_video, save_video, visualize_video
from super_gradients.training.utils.media.load_image import load_images, ImageSource, generate_loaded_image, list_images_in_folder, save_image
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.models.results import Results, DetectionResults, Result, DetectionResult
from super_gradients.training.models.predictions import Prediction, DetectionPrediction
from super_gradients.training.transforms.processing import Processing, ComposeProcessing
from super_gradients.common.abstractions.abstract_logger import get_logger


logger = get_logger(__name__)


@contextmanager
def eval_mode(model: SgModule) -> None:
    """Set a model in evaluation mode and deactivate gradient computation, undo at the end.

    :param model: The model to set in evaluation mode.
    """
    _starting_mode = model.training
    model.eval()
    with torch.no_grad():
        yield
    model.train(mode=_starting_mode)


class Pipeline(ABC):
    """An abstract base class representing a processing pipeline for a specific task.
    The pipeline includes loading images, preprocessing, prediction, and postprocessing.

    :param model:           The model used for making predictions.
    :param image_processor: A single image processor or a list of image processors for preprocessing and postprocessing the images.
    :param device:          The device on which the model will be run. Defaults to "cpu". Use "cuda" for GPU support.
    """

    def __init__(self, model: SgModule, image_processor: Union[Processing, List[Processing]], class_names: List[str], device: Optional[str] = "cpu"):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names

        if isinstance(image_processor, list):
            image_processor = ComposeProcessing(image_processor)
        self.image_processor = image_processor

    def predict_images(self, images: Union[ImageSource, List[ImageSource]], batch_size: Optional[int] = None) -> Results:
        loaded_images_generator = load_images(images)
        result_generator = self._generate_prediction_result(images=loaded_images_generator, batch_size=batch_size)
        return self._combine_results(results=list(result_generator))

    def predict_video(self, video_path: str, output_video_path: str = None, batch_size: Optional[int] = 32, visualize: Optional[bool] = False):
        """Perform inference on a video file, by processing the frames in batches.

        :param video_path:          Path to the video file.
        :param output_video_path:   Path to save the resulting video. If not specified, the output video will be saved in the same directory as the input video.
        :param batch_size:          The size of each batch.
        :param visualize:           If True, visualize the video.
        """

        video_frames, fps = load_video(file_path=video_path)

        result_generator = self._generate_prediction_result(images=video_frames, batch_size=batch_size)
        frames_with_pred = [frame_result.draw() for frame_result in tqdm(result_generator, total=len(video_frames), desc="Predicting video frames")]

        if output_video_path is None:
            directory, filename = os.path.split(video_path)
            name, ext = os.path.splitext(filename)
            output_video_path = os.path.join(directory, f"{name}_{self.model.__class__.__name__}_{ext}")

        save_video(output_path=output_video_path, frames=frames_with_pred, fps=fps)
        logger.info(f"Successfully saved video with predictions to {output_video_path}")

        if visualize:
            visualize_video(output_video_path)

    def predict_image_folder(self, image_folder_path: str, output_folder_path: str, batch_size: Optional[int] = 32):
        images_paths = list_images_in_folder(image_folder_path)
        images_generator = generate_loaded_image(images_paths)
        result_generator = self._generate_prediction_result(images=images_generator, batch_size=batch_size)

        os.makedirs(output_folder_path, exist_ok=True)
        for image_path, result in tqdm(zip(images_paths, result_generator), total=len(images_paths), desc="Predicting images"):
            output_path = os.path.join(output_folder_path, os.path.basename(image_path))
            save_image(image=result.draw(), path=output_path)

        logger.info(f"Successfully processed images from {image_folder_path}, saved with predictions to {output_folder_path}")

    def _generate_prediction_result(self, images: Iterable[np.ndarray], batch_size: Optional[int] = None) -> Iterable[Result]:
        """Run the pipeline on the images as single batch or through multiple batches.

        NOTE: A core motivation to have this function as a generator is that that way it can be used in a lazy way,
              i.e. without having to load all the images into memory.

        :param images:      Iterable of numpy arrays representing images.
        :param batch_size:  The size of each batch.
        :return:            Iterable of Results object, each containing the results of the prediction and the image.
        """
        if batch_size is None:
            yield from self._generate_prediction_result_single_batch(images)
        else:
            for batch_images in generate_batch(images, batch_size):
                yield from self._generate_prediction_result_single_batch(batch_images)

    def _generate_prediction_result_single_batch(self, images: Iterable[np.ndarray]) -> Iterable[Result]:
        """Run the pipeline and return (image, predictions). The pipeline is made of 4 steps:
            1. Load images - Loading the images into a list of numpy arrays.
            2. Preprocess - Encode the image in the shape/format expected by the model
            3. Predict - Run the model on the preprocessed image
            4. Postprocess - Decode the output of the model so that the predictions are in the shape/format of original image.

        :param images:  Iterable of numpy arrays representing images.
        :return:        Iterable of Results object, each containing the results of the prediction and the image.
        """
        self.model = self.model.to(self.device)  # Make sure the model is on the correct device, as it might have been moved after init

        # Preprocess
        preprocessed_images, processing_metadatas = [], []
        for image in images:
            preprocessed_image, processing_metadata = self.image_processor.preprocess_image(image=image.copy())
            preprocessed_images.append(preprocessed_image)
            processing_metadatas.append(processing_metadata)

        # Predict
        with eval_mode(self.model):
            torch_inputs = torch.Tensor(np.array(preprocessed_images)).to(self.device)
            model_output = self.model(torch_inputs)
            predictions = self._decode_model_output(model_output, model_input=torch_inputs)

        # Postprocess
        postprocessed_predictions = []
        for image, prediction, processing_metadata in zip(images, predictions, processing_metadatas):
            prediction = self.image_processor.postprocess_predictions(predictions=prediction, metadata=processing_metadata)
            postprocessed_predictions.append(prediction)

        # Yield results one by one
        for image, predictions in zip(images, postprocessed_predictions):
            yield self._instantiate_result(image=image, predictions=predictions)

    @abstractmethod
    def _decode_model_output(self, model_output: Union[List, Tuple, torch.Tensor], model_input: np.ndarray) -> List[Prediction]:
        """Decode the model outputs, move each prediction to numpy and store it in a Prediction object.

        :param model_output:    Direct output of the model, without any post-processing.
        :param model_input:     Model input (i.e. images after preprocessing).
        :return:                Model predictions, without any post-processing.
        """
        raise NotImplementedError

    @abstractmethod
    def _instantiate_result(self, image: np.ndarray, predictions: Prediction) -> Result:
        raise NotImplementedError

    @abstractmethod
    def _combine_results(self, results: List[Result]) -> Results:
        raise NotImplementedError


class DetectionPipeline(Pipeline):
    """Pipeline specifically designed for object detection tasks.
    The pipeline includes loading images, preprocessing, prediction, and postprocessing.

    :param model:                       The object detection model (instance of SgModule) used for making predictions.
    :param class_names:                 List of class names corresponding to the model's output classes.
    :param post_prediction_callback:    Callback function to process raw predictions from the model.
    :param image_processor:             Single image processor or a list of image processors for preprocessing and postprocessing the images.
    :param device:                      The device on which the model will be run. Defaults to "cpu". Use "cuda" for GPU support.
    """

    def __init__(
        self,
        model: SgModule,
        class_names: List[str],
        post_prediction_callback: DetectionPostPredictionCallback,
        device: Optional[str] = "cpu",
        image_processor: Optional[Processing] = None,
    ):
        super().__init__(model=model, device=device, image_processor=image_processor, class_names=class_names)
        self.post_prediction_callback = post_prediction_callback

    def _decode_model_output(self, model_output: Union[List, Tuple, torch.Tensor], model_input: np.ndarray) -> List[DetectionPrediction]:
        """Decode the model output, by applying post prediction callback. This includes NMS.

        :param model_output:    Direct output of the model, without any post-processing.
        :param model_input:     Model input (i.e. images after preprocessing).
        :return:                Predicted Bboxes.
        """
        post_nms_predictions = self.post_prediction_callback(model_output, device=self.device)

        predictions = []
        for prediction, image in zip(post_nms_predictions, model_input):
            prediction = prediction if prediction is not None else torch.zeros((0, 6), dtype=torch.float32)
            prediction = prediction.detach().cpu().numpy()
            predictions.append(
                DetectionPrediction(
                    bboxes=prediction[:, :4],
                    confidence=prediction[:, 4],
                    labels=prediction[:, 5],
                    bbox_format="xyxy",
                    image_shape=image.shape,
                )
            )

        return predictions

    def _instantiate_result(self, image: np.ndarray, predictions: DetectionPrediction) -> DetectionResult:
        return DetectionResult(image=image, predictions=predictions, class_names=self.class_names)

    def _combine_results(self, results: List[DetectionResult]) -> DetectionResults:
        return DetectionResults(results)
