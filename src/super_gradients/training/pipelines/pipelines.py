import copy
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Iterable
from contextlib import contextmanager

from super_gradients.module_interfaces import SupportsInputShapeCheck
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
    ImageClassificationPrediction,
    ImagesClassificationPrediction,
    ClassificationPrediction,
    ImageSegmentationPrediction,
    ImagesSegmentationPrediction,
    SegmentationPrediction,
    VideoSegmentationPrediction,
)
from super_gradients.training.utils.utils import generate_batch, infer_model_device, resolve_torch_device
from super_gradients.training.utils.media.video import includes_video_extension, lazy_load_video
from super_gradients.training.utils.media.image import ImageSource, check_image_typing
from super_gradients.training.utils.media.stream import WebcamStreaming
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.processing.processing import Processing, ComposeProcessing, ImagePermute
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

    :param model:               The model used for making predictions.
    :param image_processor:     A single image processor or a list of image processors for preprocessing and postprocessing the images.
    :param device:              The device on which the model will be run. If None, will run on current model device. Use "cuda" for GPU support.
    :param dtype:               Specify the dtype of the inputs. If None, will use the dtype of the model's parameters.
    :param fuse_model:          If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
    """

    def __init__(
        self,
        model: SgModule,
        image_processor: Union[Processing, List[Processing]],
        class_names: List[str],
        device: Optional[str] = None,
        fuse_model: bool = True,
        dtype: Optional[torch.dtype] = None,
        fp16: bool = True,
    ):
        model_device: torch.device = infer_model_device(model=model)
        if device:
            device: torch.device = resolve_torch_device(device=device)

        self.device: torch.device = device or model_device
        self.dtype = dtype or next(model.parameters()).dtype
        self.model = model.to(device) if device and device != model_device else model
        self.class_names = class_names

        if isinstance(image_processor, list):
            image_processor = ComposeProcessing(image_processor)

        self.image_processor = image_processor

        self.fuse_model = fuse_model  # If True, the model will be fused in the first forward pass, to make sure it gets the right input_size
        self.fp16 = fp16

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
        :param batch_size:  Maximum number of images to process at the same time.
        :return:            Results of the prediction.
        """

        if includes_video_extension(inputs):
            return self.predict_video(inputs, batch_size)
        elif check_image_typing(inputs):
            return self.predict_images(inputs, batch_size)
        else:
            raise ValueError(f"Input {inputs} not supported for prediction.")

    def predict_images(self, images: Union[ImageSource, List[ImageSource]], batch_size: Optional[int] = 32) -> Union[ImagesPredictions, ImagePrediction]:
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
        video_frames, fps, num_frames = lazy_load_video(file_path=video_path)
        result_generator = self._generate_prediction_result(images=video_frames, batch_size=batch_size)
        return self._combine_image_prediction_to_video(result_generator, fps=fps, n_images=num_frames)
        # return self._combine_image_prediction_to_video(result_generator, fps=fps, n_images=len(video_frames))

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
        # Make sure the model is on the correct device, as it might have been moved after init
        model_device: torch.device = infer_model_device(model=self.model)
        if self.device != model_device:
            self.model = self.model.to(self.device)

        images = list(images)  # We need to load all the images into memory, and to reuse it afterwards.

        # Preprocess
        preprocessed_images, processing_metadatas = [], []
        for image in images:
            preprocessed_image, processing_metadata = self.image_processor.preprocess_image(image=image.copy())
            preprocessed_images.append(preprocessed_image)
            processing_metadatas.append(processing_metadata)

        reference_shape = preprocessed_images[0].shape
        for img in preprocessed_images:
            if img.shape != reference_shape:
                raise ValueError(
                    f"Images have different shapes ({img.shape} != {reference_shape})!\n"
                    f"Either resize the images to the same size, set `skip_image_resizing=False` or pass one image at a time."
                )

        # Predict
        with eval_mode(self.model), torch.no_grad(), torch.cuda.amp.autocast(enabled=self.fp16):
            torch_inputs = torch.from_numpy(np.array(preprocessed_images)).to(self.device)
            torch_inputs = torch_inputs.to(self.dtype)

            if isinstance(self.model, SupportsInputShapeCheck):
                self.model.validate_input_shape(torch_inputs.size())

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
    def _combine_image_prediction_to_images(
        self, images_prediction_lst: Iterable[ImagePrediction], n_images: Optional[int] = None
    ) -> Union[ImagesPredictions, ImagePrediction]:
        """Instantiate an object wrapping the list of images (or ImagePrediction for single prediction)
          and the pipeline's predictions on them.

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
    :param fp16:                        If True, use mixed precision for inference.
    """

    def __init__(
        self,
        model: SgModule,
        class_names: List[str],
        post_prediction_callback: DetectionPostPredictionCallback,
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
                    labels=prediction[:, 5].astype(int),
                    bbox_format="xyxy",
                    image_shape=image.shape,
                )
            )

        return predictions

    def _instantiate_image_prediction(self, image: np.ndarray, prediction: DetectionPrediction) -> ImagePrediction:
        return ImageDetectionPrediction(image=image, prediction=prediction, class_names=self.class_names)

    def _combine_image_prediction_to_images(
        self, images_predictions: Iterable[ImageDetectionPrediction], n_images: Optional[int] = None
    ) -> Union[ImagesDetectionPrediction, ImageDetectionPrediction]:
        if n_images is not None and n_images == 1:
            # Do not show tqdm progress bar if there is only one image
            images_predictions = next(iter(images_predictions))
        else:
            images_predictions = [image_predictions for image_predictions in tqdm(images_predictions, total=n_images, desc="Predicting Images")]
            images_predictions = ImagesDetectionPrediction(_images_prediction_lst=images_predictions)

        return images_predictions

    def _combine_image_prediction_to_video(
        self, images_predictions: Iterable[ImageDetectionPrediction], fps: float, n_images: Optional[int] = None
    ) -> VideoDetectionPrediction:
        return VideoDetectionPrediction(_images_prediction_gen=images_predictions, fps=fps, n_frames=n_images)


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
        image_processor: Union[Processing, List[Processing]] = None,
        fuse_model: bool = True,
        fp16: bool = True,
    ):
        if isinstance(image_processor, list):
            image_processor = ComposeProcessing(image_processor)

        super().__init__(
            model=model,
            device=device,
            image_processor=image_processor,
            class_names=None,
            fuse_model=fuse_model,
            fp16=fp16,
        )
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
        list_of_predictions = self.post_prediction_callback(model_output)
        decoded_predictions = []
        for image_level_predictions, image in zip(list_of_predictions, model_input):
            decoded_predictions.append(
                PoseEstimationPrediction(
                    poses=image_level_predictions.poses.cpu().numpy() if torch.is_tensor(image_level_predictions.poses) else image_level_predictions.poses,
                    scores=image_level_predictions.scores.cpu().numpy() if torch.is_tensor(image_level_predictions.scores) else image_level_predictions.scores,
                    bboxes_xyxy=(
                        image_level_predictions.bboxes_xyxy.cpu().numpy()
                        if torch.is_tensor(image_level_predictions.bboxes_xyxy)
                        else image_level_predictions.bboxes_xyxy
                    ),
                    image_shape=image.shape,
                    edge_links=self.edge_links,
                    edge_colors=self.edge_colors,
                    keypoint_colors=self.keypoint_colors,
                )
            )

        return decoded_predictions

    def _instantiate_image_prediction(self, image: np.ndarray, prediction: PoseEstimationPrediction) -> ImagePrediction:
        return ImagePoseEstimationPrediction(image=image, prediction=prediction, class_names=self.class_names)

    def _combine_image_prediction_to_images(
        self, images_predictions: Iterable[PoseEstimationPrediction], n_images: Optional[int] = None
    ) -> Union[ImagesPoseEstimationPrediction, ImagePoseEstimationPrediction]:
        if n_images is not None and n_images == 1:
            # Do not show tqdm progress bar if there is only one image
            images_predictions = next(iter(images_predictions))
        else:
            images_predictions = [image_predictions for image_predictions in tqdm(images_predictions, total=n_images, desc="Predicting Images")]
            images_predictions = ImagesPoseEstimationPrediction(_images_prediction_lst=images_predictions)

        return images_predictions

    def _combine_image_prediction_to_video(
        self, images_predictions: Iterable[ImageDetectionPrediction], fps: float, n_images: Optional[int] = None
    ) -> VideoPoseEstimationPrediction:
        return VideoPoseEstimationPrediction(_images_prediction_gen=images_predictions, fps=fps, n_frames=n_images)


class ClassificationPipeline(Pipeline):
    """Pipeline specifically designed for Image Classification tasks.
    The pipeline includes loading images, preprocessing, prediction, and postprocessing.

    :param model:                       The classification model (instance of SgModule) used for making predictions.
    :param class_names:                 List of class names corresponding to the model's output classes.
    :param image_processor:             Single image processor or a list of image processors for preprocessing and postprocessing the images.
    :param device:                      The device on which the model will be run. If None, will run on current model device. Use "cuda" for GPU support.
    :param fuse_model:                  If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
    :param fp16:                        If True, use mixed precision for inference.
    """

    def __init__(
        self,
        model: SgModule,
        class_names: List[str],
        device: Optional[str] = None,
        image_processor: Union[Processing, List[Processing]] = None,
        fuse_model: bool = True,
        fp16: bool = True,
    ):
        super().__init__(
            model=model,
            device=device,
            image_processor=image_processor,
            class_names=class_names,
            fuse_model=fuse_model,
            fp16=fp16,
        )

    def _decode_model_output(self, model_output: Union[List, Tuple, torch.Tensor], model_input: np.ndarray) -> List[ClassificationPrediction]:
        """Decode the model output

        :param model_output:    Direct output of the model, without any post-processing. Tensor of shape [B, C]
        :param model_input:     Model input (i.e. images after preprocessing).
        :return:                Predicted Bboxes.
        """
        pred_scores, pred_labels = torch.max(model_output.softmax(dim=1), 1)

        pred_labels = pred_labels.detach().cpu().numpy()  # [B,1]
        pred_scores = pred_scores.detach().cpu().numpy()  # [B,1]

        predictions = list()
        for prediction, confidence, image_input in zip(pred_labels, pred_scores, model_input):
            predictions.append(ClassificationPrediction(confidence=float(confidence), label=int(prediction), image_shape=image_input.shape))
        return predictions

    def _instantiate_image_prediction(self, image: np.ndarray, prediction: ClassificationPrediction) -> ImagePrediction:
        return ImageClassificationPrediction(image=image, prediction=prediction, class_names=self.class_names)

    def _combine_image_prediction_to_images(
        self, images_predictions: Iterable[ImageClassificationPrediction], n_images: Optional[int] = None
    ) -> Union[ImagesClassificationPrediction, ImageClassificationPrediction]:
        if n_images is not None and n_images == 1:
            # Do not show tqdm progress bar if there is only one image
            images_predictions = next(iter(images_predictions))
        else:
            images_predictions = [image_predictions for image_predictions in tqdm(images_predictions, total=n_images, desc="Predicting Images")]
            images_predictions = ImagesClassificationPrediction(_images_prediction_lst=images_predictions)

        return images_predictions

    def _combine_image_prediction_to_video(
        self, images_predictions: Iterable[ImageDetectionPrediction], fps: float, n_images: Optional[int] = None
    ) -> ImagesClassificationPrediction:
        raise NotImplementedError("This feature is not available for Classification task")


class SegmentationPipeline(Pipeline):
    """Pipeline specifically designed for segmentation tasks.
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
        device: Optional[str] = None,
        image_processor: Optional[Processing] = None,
        fuse_model: bool = True,
        fp16: bool = True,
    ):
        super().__init__(model=model, device=device, image_processor=image_processor, class_names=class_names, fuse_model=fuse_model, fp16=fp16)

    def _decode_model_output(self, model_output: Union[List, Tuple, torch.Tensor], model_input: np.ndarray) -> List[SegmentationPrediction]:
        """Decode the model output, by applying post prediction callback. This includes NMS.

        :param model_output:    Direct output of the model, without any post-processing.
        :param model_input:     Model input (i.e. images after preprocessing).
        :return:                Predicted Bboxes.
        """

        if type(model_output) is tuple:
            model_output = model_output(0)

        if model_output.size(1) == 1:
            class_predication = torch.sigmoid(model_output).gt(0.5).squeeze(1).long()
        else:
            class_predication = torch.argmax(model_output, dim=1)
        class_predication = class_predication.detach().cpu().numpy()
        predictions = []
        for prediction, image in zip(class_predication, model_input):
            predictions.append(
                SegmentationPrediction(
                    segmentation_map=prediction,
                    segmentation_map_shape=prediction.shape,
                    image_shape=image.shape[-2:],
                )
            )

        return predictions

    def _instantiate_image_prediction(self, image: np.ndarray, prediction: SegmentationPrediction) -> ImagePrediction:
        return ImageSegmentationPrediction(image=image, prediction=prediction, class_names=self.class_names)

    def _combine_image_prediction_to_images(
        self, images_predictions: Iterable[ImageSegmentationPrediction], n_images: Optional[int] = None
    ) -> Union[ImagesSegmentationPrediction, ImageSegmentationPrediction]:
        if n_images is not None and n_images == 1:
            # Do not show tqdm progress bar if there is only one image
            images_predictions = next(iter(images_predictions))
        else:
            images_predictions = [image_predictions for image_predictions in tqdm(images_predictions, total=n_images, desc="Predicting Images")]
            images_predictions = ImagesSegmentationPrediction(_images_prediction_lst=images_predictions)

        return images_predictions

    def _combine_image_prediction_to_video(
        self, images_predictions: Iterable[ImageSegmentationPrediction], fps: float, n_images: Optional[int] = None
    ) -> VideoSegmentationPrediction:
        images_predictions = [image_predictions for image_predictions in tqdm(images_predictions, total=n_images, desc="Predicting Video")]
        return VideoSegmentationPrediction(_images_prediction_lst=images_predictions, fps=fps)
