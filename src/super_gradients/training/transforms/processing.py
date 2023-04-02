from typing import Tuple, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from super_gradients.training.transforms.utils import (
    _rescale_image,
    _rescale_bboxes,
    _pad_image_on_side,
    _get_center_padding_params,
    _shift_image,
    _shift_bboxes,
)


@dataclass
class ProcessingMetadata(ABC):
    """Metadata including information to postprocess a prediction."""


@dataclass
class ComposeProcessingMetadata(ProcessingMetadata):
    metadata_lst: List[Union[None, ProcessingMetadata]]


@dataclass
class DetectionPadToSizeMetadata(ProcessingMetadata):
    shift_h: float
    shift_w: float


@dataclass
class RescaleMetadata(ProcessingMetadata):
    original_size: Tuple[int, int]
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


class DetectionCenterPadding(Processing):
    """Preprocessing transform to pad image and bboxes to `output_shape` shape (H, W).
    Center padding, so that input image with bboxes located in the center of the produced image.

    Note: This transformation assume that dimensions of input image is equal or less than `output_shape`.

    :param output_shape: Output image size (H, W)
    :param pad_value:   Padding value for image
    """

    def __init__(self, output_shape: Tuple[int, int], pad_value: int):
        self.output_shape = output_shape
        self.pad_value = pad_value

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, DetectionPadToSizeMetadata]:
        shift_h, shift_w, pad_h, pad_w = _get_center_padding_params(input_size=image.shape, output_shape=self.output_shape)
        processed_image = _shift_image(image, pad_h, pad_w, self.pad_value)

        return processed_image, DetectionPadToSizeMetadata(shift_h=shift_h, shift_w=shift_w)

    def postprocess_predictions(self, predictions: np.ndarray, metadata: DetectionPadToSizeMetadata) -> np.ndarray:
        return _shift_bboxes(targets=predictions, shift_h=-metadata.shift_h, shift_w=-metadata.shift_w)


class DetectionSidePadding(Processing):
    """Preprocessing transform to pad image and bboxes to `output_shape` shape (H, W).
    Side padding, so that input image with bboxes will located on the side. Bboxes won't be affected.

    Note: This transformation assume that dimensions of input image is equal or less than `output_shape`.

    :param output_shape: Output image size (H, W)
    :param pad_value:   Padding value for image
    """

    def __init__(self, output_shape: Tuple[int, int], pad_value: int):
        self.output_shape = output_shape
        self.pad_value = pad_value

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, None]:
        processed_image = _pad_image_on_side(image, output_shape=self.output_shape, pad_val=self.pad_value)
        return processed_image, None

    def postprocess_predictions(self, predictions: np.ndarray, metadata: None) -> np.ndarray:
        return predictions


class _Rescale(Processing, ABC):
    """Resize image to given image dimensions WITHOUT preserving aspect ratio.

    :param output_shape: (H, W)
    """

    def __init__(self, output_shape: Tuple[int, int], keep_aspect_ratio: bool):
        self.output_shape = output_shape
        self.keep_aspect_ratio = keep_aspect_ratio

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, RescaleMetadata]:
        rescale_shape = self.output_shape
        scale_factor_h, scale_factor_w = rescale_shape[0] / image.shape[0], rescale_shape[1] / image.shape[1]

        if self.keep_aspect_ratio:
            scale_factor = min(scale_factor_h, scale_factor_w)
            scale_factor_h, scale_factor_w = (scale_factor, scale_factor)
            rescale_shape = (int(image.shape[0] * scale_factor_w), int(image.shape[1] * scale_factor_h))

        rescaled_image = _rescale_image(image, target_shape=rescale_shape)

        return rescaled_image, RescaleMetadata(original_size=image.shape[:2], scale_factor_h=scale_factor_h, scale_factor_w=scale_factor_w)


class DetectionRescale(_Rescale):
    def postprocess_predictions(self, predictions: np.ndarray, metadata: RescaleMetadata) -> np.ndarray:
        return _rescale_bboxes(targets=predictions, scale_factors=(1 / metadata.scale_factor_h, 1 / metadata.scale_factor_w))


class SegmentationRescale(_Rescale):
    def postprocess_predictions(self, predictions: np.ndarray, metadata: RescaleMetadata) -> np.ndarray:
        return _rescale_image(predictions, target_shape=metadata.original_size)
