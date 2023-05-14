from typing import Tuple, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from super_gradients.common.registry.registry import register_processing
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.models.predictions import Prediction, DetectionPrediction
from super_gradients.training.transforms.utils import (
    _rescale_image,
    _rescale_bboxes,
    _get_center_padding_coordinates,
    _get_bottom_right_padding_coordinates,
    _pad_image,
    _shift_bboxes,
    PaddingCoordinates,
)
from super_gradients.common.object_names import Processings


@dataclass
class ProcessingMetadata(ABC):
    """Metadata including information to postprocess a prediction."""


@dataclass
class ComposeProcessingMetadata(ProcessingMetadata):
    metadata_lst: List[Union[None, ProcessingMetadata]]


@dataclass
class DetectionPadToSizeMetadata(ProcessingMetadata):
    padding_coordinates: PaddingCoordinates


@dataclass
class RescaleMetadata(ProcessingMetadata):
    original_shape: Tuple[int, int]
    scale_factor_h: float
    scale_factor_w: float


class Processing(ABC):
    """Interface for preprocessing and postprocessing methods that are
    used to prepare images for a model and process the model's output.

    Subclasses should implement the `preprocess_image` and `postprocess_predictions`
    methods according to the specific requirements of the model and task.
    """

    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Union[None, ProcessingMetadata]]:
        """Processing an image, before feeding it to the network. Expected to be in (H, W, C) or (H, W)."""
        pass

    @abstractmethod
    def postprocess_predictions(self, predictions: Prediction, metadata: Union[None, ProcessingMetadata]) -> Prediction:
        """Postprocess the model output predictions."""
        pass


@register_processing(Processings.ComposeProcessing)
class ComposeProcessing(Processing):
    """Compose a list of Processing objects into a single Processing object."""

    def __init__(self, processings: List[Processing]):
        self.processings = processings

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, ComposeProcessingMetadata]:
        """Processing an image, before feeding it to the network."""
        processed_image, metadata_lst = image.copy(), []
        for processing in self.processings:
            processed_image, metadata = processing.preprocess_image(image=processed_image)
            metadata_lst.append(metadata)
        return processed_image, ComposeProcessingMetadata(metadata_lst=metadata_lst)

    def postprocess_predictions(self, predictions: Prediction, metadata: ComposeProcessingMetadata) -> Prediction:
        """Postprocess the model output predictions."""
        postprocessed_predictions = predictions
        for processing, metadata in zip(self.processings[::-1], metadata.metadata_lst[::-1]):
            postprocessed_predictions = processing.postprocess_predictions(postprocessed_predictions, metadata)
        return postprocessed_predictions


@register_processing(Processings.ImagePermute)
class ImagePermute(Processing):
    """Permute the image dimensions.

    :param permutation: Specify new order of dims. Default value (2, 0, 1) suitable for converting from HWC to CHW format.
    """

    def __init__(self, permutation: Tuple[int, int, int] = (2, 0, 1)):
        self.permutation = permutation

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, None]:
        processed_image = np.ascontiguousarray(image.transpose(*self.permutation))
        return processed_image, None

    def postprocess_predictions(self, predictions: Prediction, metadata: None) -> Prediction:
        return predictions


@register_processing(Processings.ReverseImageChannels)
class ReverseImageChannels(Processing):
    """Reverse the order of the image channels (RGB -> BGR or BGR -> RGB)."""

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, None]:
        """Reverse the channel order of an image.

        :param image: Image, in (H, W, C) format.
        :return:      Image with reversed channel order. (RGB if input was BGR, BGR if input was RGB)
        """

        if image.shape[2] != 3:
            raise ValueError("ReverseImageChannels expects 3 channels, got: " + str(image.shape[2]))

        processed_image = image[..., ::-1]
        return processed_image, None

    def postprocess_predictions(self, predictions: Prediction, metadata: None) -> Prediction:
        return predictions


@register_processing(Processings.StandardizeImage)
class StandardizeImage(Processing):
    """Standardize image pixel values with img/max_val

    :param max_value: Current maximum value of the image pixels. (usually 255)
    """

    def __init__(self, max_value: float = 255.0):
        super().__init__()
        self.max_value = max_value

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, None]:
        """Reverse the channel order of an image.

        :param image: Image, in (H, W, C) format.
        :return:      Image with reversed channel order. (RGB if input was BGR, BGR if input was RGB)
        """
        processed_image = (image / self.max_value).astype(np.float32)
        return processed_image, None

    def postprocess_predictions(self, predictions: Prediction, metadata: None) -> Prediction:
        return predictions


@register_processing(Processings.NormalizeImage)
class NormalizeImage(Processing):
    """Normalize an image based on means and standard deviation.

    :param mean:    Mean values for each channel.
    :param std:     Standard deviation values for each channel.
    """

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = np.array(mean).reshape((1, 1, -1)).astype(np.float32)
        self.std = np.array(std).reshape((1, 1, -1)).astype(np.float32)

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, None]:
        return (image - self.mean) / self.std, None

    def postprocess_predictions(self, predictions: Prediction, metadata: None) -> Prediction:
        return predictions


class _DetectionPadding(Processing, ABC):
    """Base class for detection padding methods. One should implement the `_get_padding_params` method to work with a custom padding method.

    Note: This transformation assume that dimensions of input image is equal or less than `output_shape`.

    :param output_shape: Output image shape (H, W)
    :param pad_value:   Padding value for image
    """

    def __init__(self, output_shape: Tuple[int, int], pad_value: int):
        self.output_shape = output_shape
        self.pad_value = pad_value

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, DetectionPadToSizeMetadata]:
        padding_coordinates = self._get_padding_params(input_shape=image.shape)
        processed_image = _pad_image(image=image, padding_coordinates=padding_coordinates, pad_value=self.pad_value)
        return processed_image, DetectionPadToSizeMetadata(padding_coordinates=padding_coordinates)

    def postprocess_predictions(self, predictions: DetectionPrediction, metadata: DetectionPadToSizeMetadata) -> DetectionPrediction:
        predictions.bboxes_xyxy = _shift_bboxes(
            targets=predictions.bboxes_xyxy,
            shift_h=-metadata.padding_coordinates.top,
            shift_w=-metadata.padding_coordinates.left,
        )
        return predictions

    @abstractmethod
    def _get_padding_params(self, input_shape: Tuple[int, int]) -> PaddingCoordinates:
        pass


@register_processing(Processings.DetectionCenterPadding)
class DetectionCenterPadding(_DetectionPadding):
    def _get_padding_params(self, input_shape: Tuple[int, int]) -> PaddingCoordinates:
        return _get_center_padding_coordinates(input_shape=input_shape, output_shape=self.output_shape)


@register_processing(Processings.DetectionBottomRightPadding)
class DetectionBottomRightPadding(_DetectionPadding):
    def _get_padding_params(self, input_shape: Tuple[int, int]) -> PaddingCoordinates:
        return _get_bottom_right_padding_coordinates(input_shape=input_shape, output_shape=self.output_shape)


class _Rescale(Processing, ABC):
    """Resize image to given image dimensions WITHOUT preserving aspect ratio.

    :param output_shape: (H, W)
    """

    def __init__(self, output_shape: Tuple[int, int]):
        self.output_shape = output_shape

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, RescaleMetadata]:

        scale_factor_h, scale_factor_w = self.output_shape[0] / image.shape[0], self.output_shape[1] / image.shape[1]
        rescaled_image = _rescale_image(image, target_shape=self.output_shape)

        return rescaled_image, RescaleMetadata(original_shape=image.shape[:2], scale_factor_h=scale_factor_h, scale_factor_w=scale_factor_w)


class _LongestMaxSizeRescale(Processing, ABC):
    """Resize image to given image dimensions WITH preserving aspect ratio.

    :param output_shape: (H, W)
    """

    def __init__(self, output_shape: Tuple[int, int]):
        self.output_shape = output_shape

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, RescaleMetadata]:
        height, width = image.shape[:2]
        scale_factor = min(self.output_shape[0] / height, self.output_shape[1] / width)

        if scale_factor != 1.0:
            new_height, new_width = round(height * scale_factor), round(width * scale_factor)
            image = _rescale_image(image, target_shape=(new_height, new_width))

        return image, RescaleMetadata(original_shape=(height, width), scale_factor_h=scale_factor, scale_factor_w=scale_factor)


@register_processing(Processings.DetectionRescale)
class DetectionRescale(_Rescale):
    def postprocess_predictions(self, predictions: DetectionPrediction, metadata: RescaleMetadata) -> DetectionPrediction:
        predictions.bboxes_xyxy = _rescale_bboxes(targets=predictions.bboxes_xyxy, scale_factors=(1 / metadata.scale_factor_h, 1 / metadata.scale_factor_w))
        return predictions


@register_processing(Processings.DetectionLongestMaxSizeRescale)
class DetectionLongestMaxSizeRescale(_LongestMaxSizeRescale):
    def postprocess_predictions(self, predictions: DetectionPrediction, metadata: RescaleMetadata) -> DetectionPrediction:
        predictions.bboxes_xyxy = _rescale_bboxes(targets=predictions.bboxes_xyxy, scale_factors=(1 / metadata.scale_factor_h, 1 / metadata.scale_factor_w))
        return predictions


def default_yolox_coco_processing_params() -> dict:
    """Processing parameters commonly used for training YoloX on COCO dataset.
    TODO: remove once we load it from the checkpoint
    """

    image_processor = ComposeProcessing(
        [
            ReverseImageChannels(),
            DetectionLongestMaxSizeRescale((640, 640)),
            DetectionBottomRightPadding((640, 640), 114),
            ImagePermute((2, 0, 1)),
        ]
    )

    params = dict(
        class_names=COCO_DETECTION_CLASSES_LIST,
        image_processor=image_processor,
        iou=0.65,
        conf=0.1,
    )
    return params


def default_ppyoloe_coco_processing_params() -> dict:
    """Processing parameters commonly used for training PPYoloE on COCO dataset.
    TODO: remove once we load it from the checkpoint
    """

    image_processor = ComposeProcessing(
        [
            ReverseImageChannels(),
            DetectionRescale(output_shape=(640, 640)),
            NormalizeImage(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            ImagePermute(permutation=(2, 0, 1)),
        ]
    )

    params = dict(
        class_names=COCO_DETECTION_CLASSES_LIST,
        image_processor=image_processor,
        iou=0.65,
        conf=0.5,
    )
    return params


def default_yolo_nas_coco_processing_params() -> dict:
    """Processing parameters commonly used for training YoloNAS on COCO dataset.
    TODO: remove once we load it from the checkpoint
    """

    image_processor = ComposeProcessing(
        [
            DetectionLongestMaxSizeRescale(output_shape=(636, 636)),
            DetectionCenterPadding(output_shape=(640, 640), pad_value=114),
            StandardizeImage(max_value=255.0),
            ImagePermute(permutation=(2, 0, 1)),
        ]
    )

    params = dict(
        class_names=COCO_DETECTION_CLASSES_LIST,
        image_processor=image_processor,
        iou=0.7,
        conf=0.25,
    )
    return params


def get_pretrained_processing_params(model_name: str, pretrained_weights: str) -> dict:
    """Get the processing parameters for a pretrained model.
    TODO: remove once we load it from the checkpoint
    """
    if pretrained_weights == "coco":
        if "yolox" in model_name:
            return default_yolox_coco_processing_params()
        elif "ppyoloe" in model_name:
            return default_ppyoloe_coco_processing_params()
        elif "yolo_nas" in model_name:
            return default_yolo_nas_coco_processing_params()
    return dict()
