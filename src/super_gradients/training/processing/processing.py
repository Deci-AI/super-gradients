from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Union

import numpy as np
from PIL import Image
import math

from super_gradients.common.object_names import Processings
from super_gradients.common.registry.registry import register_processing
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST, IMAGENET_CLASSES
from super_gradients.training.transforms.utils import (
    _rescale_image,
    _rescale_bboxes,
    _get_center_padding_coordinates,
    _get_bottom_right_padding_coordinates,
    _pad_image,
    _shift_bboxes,
    PaddingCoordinates,
    _rescale_keypoints,
    _shift_keypoints,
    _compute_scale_factor,
)
from super_gradients.training.utils.predict import Prediction, DetectionPrediction, PoseEstimationPrediction, SegmentationPrediction


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


@dataclass
class SegmentationRescaleWithPaddingMetadata(ProcessingMetadata):
    original_shape: Tuple[int, int]
    scale_factor: float
    padding_coordinates: PaddingCoordinates


@dataclass
class SegmentationResizeMetadata(ProcessingMetadata):
    original_shape: Tuple[int, int]


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


class _KeypointsPadding(Processing, ABC):
    """Base class for keypoints padding methods. One should implement the `_get_padding_params` method to work with a custom padding method.

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

    def postprocess_predictions(self, predictions: PoseEstimationPrediction, metadata: DetectionPadToSizeMetadata) -> PoseEstimationPrediction:
        predictions.poses = _shift_keypoints(
            targets=predictions.poses,
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


@register_processing(Processings.KeypointsBottomRightPadding)
class KeypointsBottomRightPadding(_KeypointsPadding):
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


@register_processing(Processings.KeypointsLongestMaxSizeRescale)
class KeypointsLongestMaxSizeRescale(_LongestMaxSizeRescale):
    def postprocess_predictions(self, predictions: PoseEstimationPrediction, metadata: RescaleMetadata) -> PoseEstimationPrediction:
        predictions.poses = _rescale_keypoints(targets=predictions.poses, scale_factors=(1 / metadata.scale_factor_h, 1 / metadata.scale_factor_w))
        return predictions


class ClassificationProcess(Processing, ABC):
    def postprocess_predictions(self, predictions: Prediction, metadata: None) -> Prediction:
        return predictions


@register_processing(Processings.Resize)
class Resize(ClassificationProcess):
    def __init__(self, size: int = 224):
        super().__init__()
        self.size = size

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, None]:
        """Resize an image.

        :param image: Image, in (H, W, C) format.
        :return:      The resized image.
        """
        image = Image.fromarray(image)
        resized_image = image.resize((self.size, self.size))
        resized_image = np.array(resized_image)

        return resized_image, None


@register_processing(Processings.CenterCrop)
class CenterCrop(ClassificationProcess):
    """
    :param size: Desired output size of the crop.
    """

    def __init__(self, size: int = 224):
        super().__init__()
        self.size = size

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, None]:
        """Crops the given image at the center.

        :param image: Image, in (H, W, C) format.
        :return:      The center cropped image.
        """
        height, width = image.shape[0], image.shape[1]

        # Calculate the start and end coordinates of the crop.
        start_x = (width - self.size) // 2
        start_y = (height - self.size) // 2
        end_x = start_x + self.size
        end_y = start_y + self.size

        cropped_image = image[start_y:end_y, start_x:end_x]
        return cropped_image, None


@register_processing(Processings.SegResizeWithPadding)
class SegResizeWithPadding(Processing, ABC):
    """Resize image to given image dimensions while preserving aspect ratio (padding might be used).

    :param output_shape:    (H, W)
    :param pad_value:    padding value (will be used if padding needed)
    """

    def __init__(self, output_shape: Tuple[int, int], pad_value: int):
        self.output_shape = output_shape
        self.pad_value = pad_value

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, SegmentationRescaleWithPaddingMetadata]:
        height, width = image.shape[:2]
        scale_factor = min(self.output_shape[0] / height, self.output_shape[1] / width)

        if scale_factor != 1.0:
            new_height, new_width = round(height * scale_factor), round(width * scale_factor)
            image = _rescale_image(image, target_shape=(new_height, new_width))

        padding_coordinates = _get_center_padding_coordinates(input_shape=image.shape, output_shape=self.output_shape)
        processed_image = _pad_image(image=image, padding_coordinates=padding_coordinates, pad_value=self.pad_value)

        return processed_image, SegmentationRescaleWithPaddingMetadata(
            original_shape=(height, width), scale_factor=scale_factor, padding_coordinates=padding_coordinates
        )

    def postprocess_predictions(self, predictions: SegmentationPrediction, metadata: SegmentationRescaleWithPaddingMetadata) -> SegmentationPrediction:
        predictions.segmentation_map = predictions.segmentation_map[
            metadata.padding_coordinates.top : predictions.segmentation_map_shape[0] - metadata.padding_coordinates.bottom,
            metadata.padding_coordinates.left : predictions.segmentation_map_shape[1] - metadata.padding_coordinates.right,
        ]
        predictions.segmentation_map = _rescale_image(predictions.segmentation_map.astype(np.uint8), target_shape=metadata.original_shape)
        predictions.segmentation_map = predictions.segmentation_map.astype(np.int64)
        return predictions


@register_processing(Processings.SegmentationRescale)
class SegmentationRescale(Processing, ABC):
    """Rescale image by scaling factor while preserving aspect ratio.
    The rescaling can be done according to scale_factor, short_size or long_size.
    If more than one argument is given, the rescaling mode is determined by this order: scale_factor, then short_size,
    then long_size.

    :param scale_factor: Rescaling is done by multiplying input size by scale_factor:
        out_size = (scale_factor * w, scale_factor * h)
    :param short_size:  Rescaling is done by determining the scale factor by the ratio short_size / min(h, w).
    :param long_size:   Rescaling is done by determining the scale factor by the ratio long_size / max(h, w).
    """

    def __init__(self, scale_factor: float, short_size: int, long_size: int):
        self.scale_factor = scale_factor
        self.short_size = short_size
        self.long_size = long_size

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, SegmentationResizeMetadata]:
        height, width = image.shape[:2]
        scale_factor = _compute_scale_factor(self.scale_factor, self.short_size, self.long_size, width, height)

        if scale_factor != 1.0:
            new_width, new_height = int(scale_factor * width), int(scale_factor * height)
            resized_image = _rescale_image(image, target_shape=(new_height, new_width))

        return resized_image, SegmentationResizeMetadata(original_shape=(height, width))

    def postprocess_predictions(self, predictions: SegmentationPrediction, metadata: SegmentationResizeMetadata) -> SegmentationPrediction:
        predictions.segmentation_map = _rescale_image(predictions.segmentation_map.astype(np.uint8), target_shape=metadata.original_shape)
        predictions.segmentation_map = predictions.segmentation_map.astype(np.int64)
        return predictions


@register_processing(Processings.SegmentationResize)
class SegmentationResize(Processing, ABC):
    """Resize image to given image dimensions.

    :param output_h:    output shape will be (output_h, output_w)
    :param output_w:    output shape will be (output_h, output_w)
    """

    def __init__(self, output_h: int, output_w: int):
        self.output_h = output_h
        self.output_w = output_w

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, SegmentationResizeMetadata]:
        height, width = image.shape[:2]
        image = _rescale_image(image, target_shape=(self.output_h, self.output_w))

        return image, SegmentationResizeMetadata(original_shape=(height, width))

    def postprocess_predictions(self, predictions: SegmentationPrediction, metadata: SegmentationResizeMetadata) -> SegmentationPrediction:
        predictions.segmentation_map = _rescale_image(predictions.segmentation_map.astype(np.uint8), target_shape=metadata.original_shape)
        predictions.segmentation_map = predictions.segmentation_map.astype(np.int64)
        return predictions


@register_processing(Processings.SegmentationPadShortToCropSize)
class SegmentationPadShortToCropSize(Processing, ABC):
    """
        Pads image to 'crop_size'.
        Should be called only after "SegRescale" or "SegRandomRescale" in augmentations pipeline.

        :param crop_size:   Tuple of (width, height) for the final crop size, if is scalar size is a square (crop_size, crop_size)
    =    :param fill_image:  Grey value to fill image padded background.
    """

    def __init__(self, crop_size: Union[float, Tuple, List], fill_image: Union[int, Tuple, List]):
        self.crop_size = crop_size
        self.fill_image = fill_image

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, DetectionPadToSizeMetadata]:
        # pad images from center symmetrically
        output_shape = max(image.shape[0], self.crop_size[0]), max(image.shape[1], self.crop_size[1])
        padding_coordinates = _get_center_padding_coordinates(input_shape=image.shape, output_shape=output_shape)
        padded_image = _pad_image(image=image, padding_coordinates=padding_coordinates, pad_value=self.fill_image)

        return padded_image, DetectionPadToSizeMetadata(padding_coordinates=padding_coordinates)

    def postprocess_predictions(self, predictions: SegmentationPrediction, metadata: DetectionPadToSizeMetadata) -> SegmentationPrediction:
        predictions.segmentation_map = predictions.segmentation_map[
            metadata.padding_coordinates.top : predictions.segmentation_map_shape[0] - metadata.padding_coordinates.bottom,
            metadata.padding_coordinates.left : predictions.segmentation_map_shape[1] - metadata.padding_coordinates.right,
        ]
        return predictions


@register_processing(Processings.SegmentationPadToDivisible)
class SegmentationPadToDivisible(Processing, ABC):
    """
        Pads image to a size divisible by the defined parameter.

        :param divisible_value:   the divisible value, new image size is an int multiplication of this number
    =    :param fill_image:  Grey value to fill image padded background.
    """

    def __init__(self, divisible_value: int, fill_image: Union[int, Tuple, List]):
        self.divisible_value = divisible_value
        self.fill_image = fill_image

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, DetectionPadToSizeMetadata]:
        h, w = image.shape
        padded_h = int(math.ceil(h / self.divisible_value) * self.divisible_value)
        padded_w = int(math.ceil(w / self.divisible_value) * self.divisible_value)

        padding_coordinates = _get_bottom_right_padding_coordinates(input_shape=image.shape, output_shape=(padded_h, padded_w))
        padded_image = _pad_image(image=image, padding_coordinates=padding_coordinates, pad_value=self.fill_image)

        return padded_image, DetectionPadToSizeMetadata(padding_coordinates=padding_coordinates)

    def postprocess_predictions(self, predictions: SegmentationPrediction, metadata: DetectionPadToSizeMetadata) -> SegmentationPrediction:
        predictions.segmentation_map = predictions.segmentation_map[
            metadata.padding_coordinates.top : predictions.segmentation_map_shape[0] - metadata.padding_coordinates.bottom,
            metadata.padding_coordinates.left : predictions.segmentation_map_shape[1] - metadata.padding_coordinates.right,
        ]
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


def default_dekr_coco_processing_params() -> dict:
    """Processing parameters commonly used for training DEKR on COCO dataset."""

    image_processor = ComposeProcessing(
        [
            ReverseImageChannels(),
            KeypointsLongestMaxSizeRescale(output_shape=(640, 640)),
            KeypointsBottomRightPadding(output_shape=(640, 640), pad_value=127),
            StandardizeImage(max_value=255.0),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ImagePermute(permutation=(2, 0, 1)),
        ]
    )

    edge_links = [
        [0, 1],
        [0, 2],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 6],
        [5, 7],
        [5, 11],
        [6, 8],
        [6, 12],
        [7, 9],
        [8, 10],
        [11, 12],
        [11, 13],
        [12, 14],
        [13, 15],
        [14, 16],
    ]

    edge_colors = [
        (214, 39, 40),  # Nose -> LeftEye
        (148, 103, 189),  # Nose -> RightEye
        (44, 160, 44),  # LeftEye -> RightEye
        (140, 86, 75),  # LeftEye -> LeftEar
        (227, 119, 194),  # RightEye -> RightEar
        (127, 127, 127),  # LeftEar -> LeftShoulder
        (188, 189, 34),  # RightEar -> RightShoulder
        (127, 127, 127),  # Shoulders
        (188, 189, 34),  # LeftShoulder -> LeftElbow
        (140, 86, 75),  # LeftTorso
        (23, 190, 207),  # RightShoulder -> RightElbow
        (227, 119, 194),  # RightTorso
        (31, 119, 180),  # LeftElbow -> LeftArm
        (255, 127, 14),  # RightElbow -> RightArm
        (148, 103, 189),  # Waist
        (255, 127, 14),  # Left Hip -> Left Knee
        (214, 39, 40),  # Right Hip -> Right Knee
        (31, 119, 180),  # Left Knee -> Left Ankle
        (44, 160, 44),  # Right Knee -> Right Ankle
    ]

    keypoint_colors = [
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
    ]
    params = dict(image_processor=image_processor, conf=0.05, edge_links=edge_links, edge_colors=edge_colors, keypoint_colors=keypoint_colors)
    return params


def default_resnet_imagenet_processing_params() -> dict:
    """Processing parameters commonly used for training resnet on Imagenet dataset."""
    image_processor = ComposeProcessing(
        [Resize(size=256), CenterCrop(size=224), NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), StandardizeImage(), ImagePermute()]
    )
    params = dict(
        class_names=IMAGENET_CLASSES,
        image_processor=image_processor,
    )
    return params


def default_ppliteseg_cityscapes_processing_params() -> dict:
    """Processing parameters commonly used for training ppliteseg on Cityscapes dataset."""
    # image_processor = ComposeProcessing(
    #     [Resize(size=256), CenterCrop(size=224), NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), StandardizeImage(), ImagePermute()]
    # )
    image_processor = ComposeProcessing(
        [
            SegResizeWithPadding(output_shape=(768, 1536), pad_value=0),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            StandardizeImage(),
            ImagePermute(),
        ]
    )
    params = dict(
        class_names=[
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ],
        image_processor=image_processor,
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

    if pretrained_weights == "coco_pose" and model_name in ("dekr_w32_no_dc", "dekr_custom"):
        return default_dekr_coco_processing_params()

    if pretrained_weights == "imagenet" and model_name == "resnet18":
        return default_resnet_imagenet_processing_params()

    if pretrained_weights == "cityscapes" and model_name == "pp_lite_t_seg75":
        return default_ppliteseg_cityscapes_processing_params()

    return dict()
