from typing import Tuple, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from super_gradients.training.transforms.utils import (
    _rescale_image,
    _rescale_bboxes,
    _get_center_padding_coordinates,
    _get_bottom_right_padding_coordinates,
    _pad_image,
    _shift_bboxes,
    PaddingCoordinates,
)


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
    def postprocess_predictions(self, predictions: np.ndarray, metadata: Union[None, ProcessingMetadata]) -> np.ndarray:
        """Postprocess the model output predictions."""
        pass


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

    def postprocess_predictions(self, predictions: np.ndarray, metadata: ComposeProcessingMetadata) -> np.ndarray:
        """Postprocess the model output predictions."""
        postprocessed_predictions = predictions
        for processing, metadata in zip(self.processings[::-1], metadata.metadata_lst[::-1]):
            postprocessed_predictions = processing.postprocess_predictions(postprocessed_predictions, metadata)
        return postprocessed_predictions


class ImagePermute(Processing):
    """Permute the image dimensions.

    :param permutation: Specify new order of dims. Default value (2, 0, 1) suitable for converting from HWC to CHW format.
    """

    def __init__(self, permutation: Tuple[int, int, int] = (2, 0, 1)):
        self.permutation = permutation

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, None]:
        processed_image = np.ascontiguousarray(image.transpose(*self.permutation))
        return processed_image, None

    def postprocess_predictions(self, predictions: np.ndarray, metadata: None) -> np.ndarray:
        return predictions


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

    def postprocess_predictions(self, predictions: np.ndarray, metadata: None) -> np.ndarray:
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

    def postprocess_predictions(self, predictions: np.ndarray, metadata: DetectionPadToSizeMetadata) -> np.ndarray:
        return _shift_bboxes(targets=predictions, shift_h=-metadata.padding_coordinates.top, shift_w=-metadata.padding_coordinates.left)

    @abstractmethod
    def _get_padding_params(self, input_shape: Tuple[int, int]) -> PaddingCoordinates:
        pass


class DetectionCenterPadding(_DetectionPadding):
    def _get_padding_params(self, input_shape: Tuple[int, int]) -> PaddingCoordinates:
        return _get_center_padding_coordinates(input_shape=input_shape, output_shape=self.output_shape)


class DetectionBottomRightPadding(_DetectionPadding):
    def _get_padding_params(self, input_shape: Tuple[int, int]) -> PaddingCoordinates:
        return _get_bottom_right_padding_coordinates(input_shape=input_shape, output_shape=self.output_shape)


class _Rescale(Processing):
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

        scale_factor = min(self.output_shape[0] / image.shape[0], self.output_shape[1] / image.shape[1])
        rescale_shape = (int(image.shape[0] * scale_factor), int(image.shape[1] * scale_factor))
        rescaled_image = _rescale_image(image, target_shape=rescale_shape)

        return rescaled_image, RescaleMetadata(original_shape=image.shape[:2], scale_factor_h=scale_factor, scale_factor_w=scale_factor)


class DetectionRescale(_Rescale):
    def postprocess_predictions(self, predictions: np.ndarray, metadata: RescaleMetadata) -> np.ndarray:
        return _rescale_bboxes(targets=predictions, scale_factors=(1 / metadata.scale_factor_h, 1 / metadata.scale_factor_w))


class DetectionLongestMaxSizeRescale(_LongestMaxSizeRescale):
    def postprocess_predictions(self, predictions: np.ndarray, metadata: RescaleMetadata) -> np.ndarray:
        return _rescale_bboxes(targets=predictions, scale_factors=(1 / metadata.scale_factor_h, 1 / metadata.scale_factor_w))


class SegmentationRescale(_Rescale):
    def postprocess_predictions(self, predictions: np.ndarray, metadata: RescaleMetadata) -> np.ndarray:
        return _rescale_image(predictions, target_shape=metadata.original_shape)
