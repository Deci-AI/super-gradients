from typing import Tuple, List, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel

import numpy as np

from super_gradients.training.transforms.utils import (
    _rescale_image,
    _rescale_bboxes,
    _shift_image,
    _shift_bboxes,
    _rescale_and_pad_to_size,
    _rescale_xyxy_bboxes,
    _get_shift_params,
)


class ProcessingMetadata(BaseModel, ABC):
    """Metadata including information to postprocess a prediction."""


class ComposeProcessingMetadata(ProcessingMetadata):
    metadata_lst: List[Union[ProcessingMetadata, None]]


class DetectionPadToSizeMetadata(ProcessingMetadata):
    shift_w: float
    shift_h: float


class RescaleMetadata(ProcessingMetadata):
    original_size: Tuple[int, int]
    sy: float
    sx: float


class DetectionPaddedRescaleMetadata(ProcessingMetadata):
    r: float


class Processing(ABC):
    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Union[ProcessingMetadata, None]]:
        """Processing an image, before feeding it to the network."""
        pass

    @abstractmethod
    def postprocess_predictions(self, predictions: np.ndarray, metadata: Union[ProcessingMetadata, None]) -> np.ndarray:
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


class DetectionPaddedRescale(Processing):
    """Apply padding rescaling to image and bboxes to `output_size` shape (rows, cols).

    :param output_size: Target input dimension.
    :param swap:        Image axis's to be rearranged.
    :param pad_value:   Padding value for image.
    """

    def __init__(self, output_size: Tuple[int, int], swap: Tuple[int, ...] = (2, 0, 1), pad_value: int = 114):
        self.output_size = output_size
        self.swap = swap
        self.pad_value = pad_value

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, DetectionPaddedRescaleMetadata]:
        rescaled_image, r = _rescale_and_pad_to_size(image=image, output_size=self.output_size, swap=self.swap, pad_val=self.pad_value)
        return rescaled_image, DetectionPaddedRescaleMetadata(r=r)

    def postprocess_predictions(self, predictions: np.array, metadata=DetectionPaddedRescaleMetadata) -> np.array:
        return _rescale_xyxy_bboxes(targets=predictions, r=1 / metadata.r)


class DetectionPadToSize(Processing):
    """Preprocessing transform to pad image and bboxes to `output_size` shape (rows, cols).
    Center padding, so that input image with bboxes located in the center of the produced image.

    Note: This transformation assume that dimensions of input image is equal or less than `output_size`.

    :param output_size: Output image size (rows, cols)
    :param pad_value:   Padding value for image
    """

    def __init__(self, output_size: Tuple[int, int], pad_value: int):
        self.output_size = output_size
        self.pad_value = pad_value

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, DetectionPadToSizeMetadata]:
        shift_h, shift_w, pad_h, pad_w = _get_shift_params(original_size=image.shape, output_size=self.output_size)
        processed_image = _shift_image(image, pad_h, pad_w, self.pad_value)

        return processed_image, DetectionPadToSizeMetadata(shift_h=shift_h, shift_w=shift_w)

    def postprocess_predictions(self, predictions: np.ndarray, metadata: DetectionPadToSizeMetadata) -> np.ndarray:
        return _shift_bboxes(targets=predictions, shift_w=-metadata.shift_w, shift_h=-metadata.shift_h)


class _Rescale(Processing, ABC):
    """Resize image to given image dimensions without preserving aspect ratio.

    :param output_shape: (rows, cols)
    """

    def __init__(self, output_shape: Tuple[int, int]):
        self.output_shape = output_shape

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, RescaleMetadata]:
        sy, sx = self.output_shape[0] / image.shape[0], self.output_shape[1] / image.shape[1]
        rescaled_image = _rescale_image(image, target_shape=self.output_shape)

        return rescaled_image, RescaleMetadata(original_size=image.shape[:2], sy=sy, sx=sx)


class DetectionRescale(_Rescale):
    def postprocess_predictions(self, predictions: np.ndarray, metadata: RescaleMetadata) -> np.ndarray:
        return _rescale_bboxes(targets=predictions, scale_factors=(1 / metadata.sy, 1 / metadata.sx))


class SegmentationRescale(_Rescale):
    def postprocess_predictions(self, predictions: np.ndarray, metadata: RescaleMetadata) -> np.ndarray:
        return _rescale_image(predictions, target_shape=metadata.original_size)
