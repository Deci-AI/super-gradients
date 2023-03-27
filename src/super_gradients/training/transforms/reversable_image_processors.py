from typing import Union, Tuple, Dict, Any
from abc import ABC, abstractmethod

import cv2
import numpy as np

from super_gradients.training.utils.detection_utils import xyxy2cxcywh, cxcywh2xyxy


class ReversibleImageProcessor(ABC):
    """Abstract base class for reversible transforms.
    This comes handy when you want to apply a transform, and then undo that transform afterwards.

    To use such a transform, you need to first calibrate the instance to an image.
    This will save the useful information to later on apply the transform and/or reverse the transform.
    Then, any of its processing method will be applied according to the calibrated image.
    """

    def __init__(self):
        self._state: Union[Dict, None] = None

    @property
    def state(self) -> dict:
        """Wrapper around the state of the transform, in order to make sure that no transformation is called before the state is set (through `calibrate`)."""
        if self._state is None:
            raise RuntimeError(f"`calibrate` must be applied first before calling other methods if {self.__name__}.")
        return self._state

    @state.setter
    def state(self, value: Any):
        self._state = value

    @abstractmethod
    def calibrate(self, image: np.ndarray) -> None:
        """Calibrate the state of the reversible image processor. This state will be used in subsequent transforms, until this instance is calibrated again."""
        raise NotImplementedError

    @abstractmethod
    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """Apply the transform to the image.

        :param image: Original image
        :return:      Transformed image
        """
        raise NotImplementedError

    @abstractmethod
    def apply_reverse_to_image(self, image: np.ndarray) -> np.ndarray:
        """Reverse the transform to the image.

        :param image: Transformed image
        :return:      Original image
        """
        raise NotImplementedError


class ReversibleDetectionProcessor(ReversibleImageProcessor):
    """Abstract base class for reversible transforms. The solution we chose is to store a "state" attribute when transforming an image.
    This attribute can be used to apply the same transform on targets
    """

    @abstractmethod
    def apply_to_targets(self, targets: np.array) -> np.array:
        """Reverse transform on bboxes.

        :param targets:  Transformed Bboxes, of shape (N, 5), in format [x1, y1, x2, y2, class_id, ...]
        :return:         Original Bboxes, of shape (N, 5), in format [x1, y1, x2, y2, class_id, ...]
        """
        raise NotImplementedError

    @abstractmethod
    def apply_reverse_to_targets(self, targets: np.array) -> np.array:
        """Reverse transform on bboxes.

        :param targets:  Transformed Bboxes, of shape (N, 5), in format [x1, y1, x2, y2, class_id, ...]
        :return:         Original Bboxes, of shape (N, 5), in format [x1, y1, x2, y2, class_id, ...]
        """
        raise NotImplementedError


class ReversibleDetectionRescale(ReversibleDetectionProcessor):
    """
    Resize image and bounding boxes to given image dimensions without preserving aspect ratio

    :param output_shape: (rows, cols)
    """

    def __init__(self, output_shape: Tuple[int, int]):
        super().__init__()
        self.output_shape = output_shape

    def calibrate(self, image: np.ndarray) -> None:
        original_size = image.shape
        sy, sx = self.output_shape[0] / original_size[0], self.output_shape[1] / original_size[1]
        self.state = {"original_size": original_size, "scale_factors": (sy, sx)}

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        output_shape = self.output_shape
        return _rescale_image(image, target_shape=output_shape)

    def apply_reverse_to_image(self, image: np.ndarray) -> np.ndarray:
        original_size = self.state["original_size"]
        return _rescale_image(image=image, target_shape=original_size)

    def apply_to_targets(self, targets: np.array) -> np.array:
        sy, sx = self.state["scale_factors"]
        return _rescale_target(targets=targets, scale_factors=(sy, sx))

    def apply_reverse_to_targets(self, targets: np.array) -> np.array:
        sy, sx = self.state["scale_factors"]
        return _rescale_target(targets=targets, scale_factors=(1 / sy, 1 / sx))


class ReversibleDetectionPadToSize(ReversibleDetectionProcessor):
    """Preprocessing transform to pad image and bboxes to `target_size` shape (rows, cols).
    Transform does center padding, so that input image with bboxes located in the center of the produced image.

    Note: This transformation assume that dimensions of input image is equal or less than `output_size`.


    :param output_size: Output image size (rows, cols)
    :param pad_value: Padding value for image
    """

    def __init__(self, output_size: Tuple[int, int], pad_value: int):
        super().__init__()
        self.output_size = output_size
        self.pad_value = pad_value

    def calibrate(self, image: np.ndarray) -> None:
        original_size = image.shape

        pad_h, pad_w = self.output_size[0] - original_size[0], self.output_size[1] - original_size[1]
        shift_h, shift_w = pad_h // 2, pad_w // 2
        pad_h = (shift_h, pad_h - shift_h)
        pad_w = (shift_w, pad_w - shift_w)
        self.state = {"original_size": original_size, "shift_w": shift_w, "shift_h": shift_h, "pad_h": pad_h, "pad_w": pad_w}

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        pad_h, pad_w = self.state["pad_h"], self.state["pad_w"]

        return np.pad(image, (pad_h, pad_w, (0, 0)), mode="constant", constant_values=self.pad_value)

    def apply_reverse_to_image(self, image: np.ndarray) -> np.ndarray:
        start_h, end_h = self.state["pad_h"]
        start_w, end_w = self.state["pad_w"]
        original_size = self.state["original_size"]

        return image[start_h : original_size[0] + start_h, start_w : original_size[1] + start_w]

    def apply_to_targets(self, targets: np.array) -> np.array:
        shift_w, shift_h = self.state["shift_w"], self.state["shift_h"]

        return _translate_targets(targets=targets, shift_w=shift_w, shift_h=shift_h)

    def apply_reverse_to_targets(self, targets: np.array) -> np.array:
        shift_w, shift_h = self.state["shift_w"], self.state["shift_h"]

        return _translate_targets(targets=targets, shift_w=-shift_w, shift_h=-shift_h)


class ReversibleDetectionPaddedRescale(ReversibleDetectionProcessor):
    """Apply padding rescaling to image and bboxes to `target_size` shape (rows, cols).

    :param target_size: Target input dimension.
    :param swap:        Image axis's to be rearranged.
    :param pad_value:   Padding value for image.
    """

    def __init__(self, target_size: Tuple[int, int], swap: Tuple[int, ...] = (2, 0, 1), pad_value: int = 114):
        super().__init__()
        self.target_size = target_size
        self.swap = swap
        self.pad_value = pad_value

    def calibrate(self, image: np.ndarray) -> None:
        r = min(self.target_size[0] / image.shape[0], self.target_size[1] / image.shape[1])
        self.state = {"original_size": image.shape, "r": r}

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        r = self.state["r"]
        return _rescale_and_pad_to_size(image=image, target_size=self.target_size, r=r, pad_val=self.pad_value, swap=self.swap)

    def apply_reverse_to_image(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_to_targets(self, targets: np.array) -> np.array:
        r = self.state["r"]
        return _rescale_xyxy_target(targets=targets, r=r)

    def apply_reverse_to_targets(self, targets: np.array) -> np.array:
        r = 1 / self.state["r"]
        return _rescale_xyxy_target(targets=targets, r=r)


def _compute_input_output_size_ratio(input_size: Tuple[int, int], output_size: Tuple[int, int]) -> float:
    return min(output_size[0] / input_size[0], output_size[1] / input_size[1])


def _rescale_target(targets: np.array, scale_factors: Tuple[float, float]) -> np.array:
    """Rescale targets to given scale factors."""
    sy, sx = scale_factors
    targets = targets.astype(np.float32, copy=True) if len(targets) > 0 else np.zeros((0, 5), dtype=np.float32)
    targets[:, 0:4] *= np.array([[sx, sy, sx, sy]], dtype=targets.dtype)
    return targets


def _rescale_image(image: np.ndarray, target_shape: Tuple[float, float]) -> np.ndarray:
    """Rescale image to target_shape, without preserving aspect ratio."""
    return cv2.resize(image, dsize=(int(target_shape[1]), int(target_shape[0])), interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def _translate_targets(targets: np.array, shift_w: float, shift_h: float) -> np.array:
    """Translate bboxes with respect to padding values.

    :param targets:  Bboxes to transform of shape (N, 5), in format [x1, y1, x2, y2, class_id, ...]
    :param shift_w:  shift width in pixels
    :param shift_h:  shift height in pixels
    :return:         Bboxes to transform of shape (N, 5), in format [x1, y1, x2, y2, class_id, ...]
    """
    targets = targets.copy() if len(targets) > 0 else np.zeros((0, 5), dtype=np.float32)
    boxes, labels = targets[:, :4], targets[:, 4:]
    boxes[:, [0, 2]] += shift_w
    boxes[:, [1, 3]] += shift_h
    return np.concatenate((boxes, labels), 1)


def _rescale_xyxy_target(targets: np.array, r: float) -> np.array:
    """Scale targets to given scale factors.

    :param targets:  Targets to rescale, shape (batch_size, 6)
    :param r:        SegRescale coefficient that was applied to the image
    :return:         Rescaled targets, shape (batch_size, 6)
    """
    targets = targets.copy()
    boxes, labels = targets[:, :4], targets[:, 4]
    boxes = xyxy2cxcywh(boxes)
    boxes *= r
    boxes = cxcywh2xyxy(boxes)
    return np.concatenate((boxes, labels[:, np.newaxis]), 1)


def _rescale_and_pad_to_size(image: np.ndarray, target_size: Tuple[int, int], r: float, swap: Tuple[int] = (2, 0, 1), pad_val: int = 114) -> np.ndarray:
    """
    Rescales image according to minimum ratio between the target height /image height, target width / image width,
    and pads the image to the target size.

    :param image:       Image to be rescaled
    :param target_size: Target size
    :param r:           Rescale coefficient
    :param swap:        Axis's to be rearranged.
    :param pad_val:     Value to use for padding
    :return:            Rescaled image according to ratio r and padded to fit target_size.
    """
    if len(image.shape) == 3:
        padded_image = np.ones((target_size[0], target_size[1], image.shape[-1]), dtype=np.uint8) * pad_val
    else:
        padded_image = np.ones(target_size, dtype=np.uint8) * pad_val

    target_shape = (int(image.shape[0] * r), int(image.shape[2] * r))
    resized_image = _rescale_image(image=image, target_shape=target_shape)
    padded_image[: target_shape[0], : target_shape[1]] = resized_image

    padded_image = padded_image.transpose(swap)
    padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
    return padded_image
