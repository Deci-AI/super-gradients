import copy
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Iterable
from contextlib import contextmanager
from tqdm import tqdm

import numpy as np
import torch

from super_gradients.training.utils.predict import (
    ImagePoseEstimationPrediction,
    ImagesPoseEstimationPrediction,
    VideoPoseEstimationPrediction,
    ImagesDetectionPrediction,
    VideoDetectionPrediction,
    ImagePrediction,
    ImageDetectionPrediction,
    ImagesPredictions,
    VideoPredictions,
    Prediction,
    DetectionPrediction,
    PoseEstimationPrediction,
)
from super_gradients.training.utils.utils import generate_batch
from super_gradients.training.utils.media.video import load_video, includes_video_extension
from super_gradients.training.utils.media.image import ImageSource, check_image_typing
from super_gradients.training.utils.media.stream import WebcamStreaming
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.processing.processing import Processing, ComposeProcessing
from super_gradients.common.abstractions.abstract_logger import get_logger


logger = get_logger(__name__)


@contextmanager
def eval_mode(model: SgModule) -> None:
    """Set a model in evaluation mode, undo at the end.

    :param model: The model to set in evaluation mode.
    """
    _starting_mode = model.training
    model.eval()
    yield
    model.train(mode=_starting_mode)


class Pipeline(ABC):
    """An abstract base class representing a processing pipeline for a specific task.
    The pipeline includes loading images, preprocessing, prediction, and postprocessing.

    :param model:           The model used for making predictions.
    :param image_processor: A single image processor or a list of image processors for preprocessing and postprocessing the images.
    :param device:          The device on which the model will be run. If None, will run on current model device. Use "cuda" for GPU support.
    :param fuse_model:                  If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
    """

    def __init__(
        self,
        model: SgModule,
        image_processor: Union[Processing, List[Processing]],
        class_names: List[str],
        device: Optional[str] = None,
        fuse_model: bool = True,
    ):
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.class_names = class_names

        if isinstance(image_processor, list):
            image_processor = ComposeProcessing(image_processor)
        self.image_processor = image_processor

        self.fuse_model = fuse_model  # If True, the model will be fused in the first forward pass, to make sure it gets the right input_size

    def _fuse_model(self, input_example: torch.Tensor):
        logger.info("Fusing some of the model's layers. If this takes too much memory, you can deactivate it by setting `fuse_model=False`")
        self.model = copy.deepcopy(self.model)
        self.model.eval()
        self.model.prep_model_for_conversion(input_size=input_example.shape[-2:])
        self.fuse_model = False

    def __call__(self, inputs: Union[str, ImageSource, List[ImageSource]], batch_size: Optional[int] = 32) -> ImagesPredictions:
        """Predict an image or a list of images.

        Supported types include:
            - str:              A string representing either a video, an image or an URL.
            - numpy.ndarray:    A numpy array representing the image
            - torch.Tensor:     A PyTorch tensor representing the image
            - PIL.Image.Image:  A PIL Image object
            - List:             A list of images of any of the above image types (list of videos not supported).

        :param inputs:      inputs to the model, which can be any of the above-mentioned types.
        :param batch_size:  Number of images to be processed at the same time.
        :return:            Results of the prediction.
        """

        if includes_video_extension(inputs):
            return self.predict_video(inputs, batch_size)
        elif check_image_typing(inputs):
            return self.predict_images(inputs, batch_size)
        else:
            raise ValueError(f"Input {inputs} not supported for prediction.")

    def predict_images(self, images: Union[ImageSource, List[ImageSource]], batch_size: Optional[int] = 32) -> ImagesPredictions:
        """Predict an image or a list of images.

        :param images:      Images to predict.
        :param batch_size:  The size of each batch.
        :return:            Results of the prediction.
        """
        from super_gradients.training.utils.media.image import load_images

        images = load_images(images)
        result_generator = self._generate_prediction_result(images=images, batch_size=batch_size)
        return self._combine_image_prediction_to_images(result_generator, n_images=len(images))

    def predict_video(self, video_path: str, batch_size: Optional[int] = 32) -> VideoPredictions:
        """Predict on a video file, by processing the frames in batches.

        :param video_path:  Path to the video file.
        :param batch_size:  The size of each batch.
        :return:            Results of the prediction.
        """
        video_frames, fps = load_video(file_path=video_path)
        result_generator = self._generate_prediction_result(images=video_frames, batch_size=batch_size)
        return self._combine_image_prediction_to_video(result_generator, fps=fps, n_images=len(video_frames))

    def predict_webcam(self) -> None:
        """Predict using webcam"""

        def _draw_predictions(frame: np.ndarray) -> np.ndarray:
            """Draw the predictions on a single frame from the stream."""
            frame_prediction = next(iter(self._generate_prediction_result(images=[frame])))
            return frame_prediction.draw()

        video_streaming = WebcamStreaming(frame_processing_fn=_draw_predictions, fps_update_frequency=1)
        video_streaming.run()

    def _generate_prediction_result(self, images: Iterable[np.ndarray], batch_size: Optional[int] = None) -> Iterable[ImagePrediction]:
        """Run the pipeline on the images as single batch or through multiple batches.

        NOTE: A core motivation to have this function as a generator is that it can be used in a lazy way (if images is generator itself),
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

    def _generate_prediction_result_single_batch(self, images: Iterable[np.ndarray]) -> Iterable[ImagePrediction]:
        """Run the pipeline on images. The pipeline is made of 4 steps:
            1. Load images - Loading the images into a list of numpy arrays.
            2. Preprocess - Encode the image in the shape/format expected by the model
            3. Predict - Run the model on the preprocessed image
            4. Postprocess - Decode the output of the model so that the predictions are in the shape/format of original image.

        :param images:  Iterable of numpy arrays representing images.
        :return:        Iterable of Results object, each containing the results of the prediction and the image.
        """
        images = list(images)  # We need to load all the images into memory, and to reuse it afterwards.
        self.model = self.model.to(self.device)  # Make sure the model is on the correct device, as it might have been moved after init

        # Preprocess
        preprocessed_images, processing_metadatas = [], []
        for image in images:
            preprocessed_image, processing_metadata = self.image_processor.preprocess_image(image=image.copy())
            preprocessed_images.append(preprocessed_image)
            processing_metadatas.append(processing_metadata)

        # Predict
        with eval_mode(self.model), torch.no_grad(), torch.cuda.amp.autocast():
            torch_inputs = torch.from_numpy(np.array(preprocessed_images)).to(self.device)
            if self.fuse_model:
                self._fuse_model(torch_inputs)
            model_output = self.model(torch_inputs)
            predictions = self._decode_model_output(model_output, model_input=torch_inputs)

        # Postprocess
        postprocessed_predictions = []
        for image, prediction, processing_metadata in zip(images, predictions, processing_metadatas):
            prediction = self.image_processor.postprocess_predictions(predictions=prediction, metadata=processing_metadata)
            postprocessed_predictions.append(prediction)

        # Yield results one by one
        for image, prediction in zip(images, postprocessed_predictions):
            yield self._instantiate_image_prediction(image=image, prediction=prediction)

    @abstractmethod
    def _decode_model_output(self, model_output: Union[List, Tuple, torch.Tensor], model_input: np.ndarray) -> List[Prediction]:
        """Decode the model outputs, move each prediction to numpy and store it in a Prediction object.

        :param model_output:    Direct output of the model, without any post-processing.
        :param model_input:     Model input (i.e. images after preprocessing).
        :return:                Model predictions, without any post-processing.
        """
        raise NotImplementedError

    @abstractmethod
    def _instantiate_image_prediction(self, image: np.ndarray, prediction: Prediction) -> ImagePrediction:
        """Instantiate an object wrapping an image and the pipeline's prediction.

        :param image:       Image to predict.
        :param prediction:  Model prediction on that image.
        :return:            Object wrapping an image and the pipeline's prediction.
        """
        raise NotImplementedError

    @abstractmethod
    def _combine_image_prediction_to_images(self, images_prediction_lst: Iterable[ImagePrediction], n_images: Optional[int] = None) -> ImagesPredictions:
        """Instantiate an object wrapping the list of images and the pipeline's predictions on them.

        :param images_prediction_lst:   List of image predictions.
        :param n_images:                (Optional) Number of images in the list. This used for tqdm progress bar to work with iterables, but is not required.
        :return:                        Object wrapping the list of image predictions.
        """
        raise NotImplementedError

    @abstractmethod
    def _combine_image_prediction_to_video(
        self, images_prediction_lst: Iterable[ImagePrediction], fps: float, n_images: Optional[int] = None
    ) -> VideoPredictions:
        """Instantiate an object holding the video frames and the pipeline's predictions on it.

        :param images_prediction_lst:   List of image predictions.
        :param fps:                     Frames per second.
        :param n_images:                (Optional) Number of images in the list. This used for tqdm progress bar to work with iterables, but is not required.
        :return:                        Object wrapping the list of image predictions as a Video.
        """
        raise NotImplementedError


class DetectionPipeline(Pipeline):
    """Pipeline specifically designed for object detection tasks.
    The pipeline includes loading images, preprocessing, prediction, and postprocessing.

    :param model:                       The object detection model (instance of SgModule) used for making predictions.
    :param class_names:                 List of class names corresponding to the model's output classes.
    :param post_prediction_callback:    Callback function to process raw predictions from the model.
    :param image_processor:             Single image processor or a list of image processors for preprocessing and postprocessing the images.
    :param device:                      The device on which the model will be run. If None, will run on current model device. Use "cuda" for GPU support.
    :param fuse_model:                  If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
    """

    def __init__(
        self,
        model: SgModule,
        class_names: List[str],
        post_prediction_callback: DetectionPostPredictionCallback,
        device: Optional[str] = None,
        image_processor: Optional[Processing] = None,
        fuse_model: bool = True,
    ):
        super().__init__(model=model, device=device, image_processor=image_processor, class_names=class_names, fuse_model=fuse_model)
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

    def _instantiate_image_prediction(self, image: np.ndarray, prediction: DetectionPrediction) -> ImagePrediction:
        return ImageDetectionPrediction(image=image, prediction=prediction, class_names=self.class_names)

    def _combine_image_prediction_to_images(
        self, images_predictions: Iterable[ImageDetectionPrediction], n_images: Optional[int] = None
    ) -> ImagesDetectionPrediction:
        if n_images is not None and n_images == 1:
            # Do not show tqdm progress bar if there is only one image
            images_predictions = [next(iter(images_predictions))]
        else:
            images_predictions = [image_predictions for image_predictions in tqdm(images_predictions, total=n_images, desc="Predicting Images")]

        return ImagesDetectionPrediction(_images_prediction_lst=images_predictions)

    def _combine_image_prediction_to_video(
        self, images_predictions: Iterable[ImageDetectionPrediction], fps: float, n_images: Optional[int] = None
    ) -> VideoDetectionPrediction:
        images_predictions = [image_predictions for image_predictions in tqdm(images_predictions, total=n_images, desc="Predicting Video")]
        return VideoDetectionPrediction(_images_prediction_lst=images_predictions, fps=fps)


class PoseEstimationPipeline(Pipeline):
    """Pipeline specifically designed for pose estimation tasks.
    The pipeline includes loading images, preprocessing, prediction, and postprocessing.

    :param model:                       The object detection model (instance of SgModule) used for making predictions.
    :param post_prediction_callback:    Callback function to process raw predictions from the model.
    :param image_processor:             Single image processor or a list of image processors for preprocessing and postprocessing the images.
    :param device:                      The device on which the model will be run. If None, will run on current model device. Use "cuda" for GPU support.
    :param fuse_model:                  If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
    """

    def __init__(
        self,
        model: SgModule,
        edge_links: Union[np.ndarray, List[Tuple[int, int]]],
        edge_colors: Union[np.ndarray, List[Tuple[int, int, int]]],
        keypoint_colors: Union[np.ndarray, List[Tuple[int, int, int]]],
        post_prediction_callback,
        device: Optional[str] = None,
        image_processor: Optional[Processing] = None,
        fuse_model: bool = True,
    ):
        super().__init__(model=model, device=device, image_processor=image_processor, class_names=None, fuse_model=fuse_model)
        self.post_prediction_callback = post_prediction_callback
        self.edge_links = np.asarray(edge_links, dtype=int)
        self.edge_colors = np.asarray(edge_colors, dtype=int)
        self.keypoint_colors = np.asarray(keypoint_colors, dtype=int)

    def _decode_model_output(self, model_output: Union[List, Tuple, torch.Tensor], model_input: np.ndarray) -> List[PoseEstimationPrediction]:
        """Decode the model output, by applying post prediction callback. This includes NMS.

        :param model_output:    Direct output of the model, without any post-processing.
        :param model_input:     Model input (i.e. images after preprocessing).
        :return:                Predicted Bboxes.
        """
        all_poses, all_scores = self.post_prediction_callback(model_output)

        predictions = []
        for poses, scores, image in zip(all_poses, all_scores, model_input):
            predictions.append(
                PoseEstimationPrediction(
                    poses=poses,
                    scores=scores,
                    image_shape=image.shape,
                    edge_links=self.edge_links,
                    edge_colors=self.edge_colors,
                    keypoint_colors=self.keypoint_colors,
                )
            )

        return predictions

    def _instantiate_image_prediction(self, image: np.ndarray, prediction: PoseEstimationPrediction) -> ImagePrediction:
        return ImagePoseEstimationPrediction(image=image, prediction=prediction, class_names=self.class_names)

    def _combine_image_prediction_to_images(
        self, images_predictions: Iterable[PoseEstimationPrediction], n_images: Optional[int] = None
    ) -> ImagesPoseEstimationPrediction:
        if n_images is not None and n_images == 1:
            # Do not show tqdm progress bar if there is only one image
            images_predictions = [next(iter(images_predictions))]
        else:
            images_predictions = [image_predictions for image_predictions in tqdm(images_predictions, total=n_images, desc="Predicting Images")]

        return ImagesPoseEstimationPrediction(_images_prediction_lst=images_predictions)

    def _combine_image_prediction_to_video(
        self, images_predictions: Iterable[ImageDetectionPrediction], fps: float, n_images: Optional[int] = None
    ) -> VideoPoseEstimationPrediction:
        images_predictions = [image_predictions for image_predictions in tqdm(images_predictions, total=n_images, desc="Predicting Video")]
        return VideoPoseEstimationPrediction(_images_prediction_lst=images_predictions, fps=fps)
