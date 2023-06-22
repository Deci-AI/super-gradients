from typing import Tuple
from dataclasses import dataclass
import cv2

import numpy as np

from super_gradients.training.utils.detection_utils import xyxy2cxcywh, cxcywh2xyxy


@dataclass
class PaddingCoordinates:
    top: int
    bottom: int
    left: int
    right: int


def _rescale_image(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Rescale image to target_shape, without preserving aspect ratio.

    :param image:           Image to rescale. (H, W, C) or (H, W).
    :param target_shape:    Target shape to rescale to (H, W).
    :return:                Rescaled image.
    """
    height, width = target_shape[:2]
    return cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_LINEAR)


def _rescale_bboxes(targets: np.ndarray, scale_factors: Tuple[float, float]) -> np.ndarray:
    """Rescale bboxes to given scale factors, without preserving aspect ratio.

    :param targets:         Targets to rescale (N, 4+), where target[:, :4] is the bounding box coordinates.
    :param scale_factors:   Tuple of (scale_factor_h, scale_factor_w) scale factors to rescale to.
    :return:                Rescaled targets.
    """

    targets = targets.astype(np.float32, copy=True)

    sy, sx = scale_factors
    targets[:, :4] *= np.array([[sx, sy, sx, sy]], dtype=targets.dtype)
    return targets


def _rescale_keypoints(targets: np.ndarray, scale_factors: Tuple[float, float]) -> np.ndarray:
    """Rescale keypoints to given scale factors, without preserving aspect ratio.

    :param targets:         Array of keypoints to rescale. Can have arbitrary shape [N,2], [N,K,2], etc.
                            Last dimension encodes XY coordinates: target[..., 0] is the X coordinates and
                            targets[..., 1] is the Y coordinate.
    :param scale_factors:   Tuple of (scale_factor_h, scale_factor_w) scale factors to rescale to.
    :return:                Rescaled targets.
    """

    targets = targets.astype(np.float32, copy=True)

    sy, sx = scale_factors
    targets[..., 0] *= sx
    targets[..., 1] *= sy
    return targets


def _get_center_padding_coordinates(input_shape: Tuple[int, int], output_shape: Tuple[int, int]) -> PaddingCoordinates:
    """Get parameters for padding an image to given output shape, in center mode.

    :param input_shape:  Shape of the input image.
    :param output_shape: Shape to resize to.
    :return:             Padding parameters.
    """
    pad_height, pad_width = output_shape[0] - input_shape[0], output_shape[1] - input_shape[1]

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return PaddingCoordinates(top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right)


def _get_bottom_right_padding_coordinates(input_shape: Tuple[int, int], output_shape: Tuple[int, int]) -> PaddingCoordinates:
    """Get parameters for padding an image to given output shape, in bottom right mode
    (i.e. image will be at top-left while bottom-right corner will be padded).

    :param input_shape:  Shape of the input image.
    :param output_shape: Shape to resize to.
    :return:             Padding parameters.
    """
    pad_height, pad_width = output_shape[0] - input_shape[0], output_shape[1] - input_shape[1]
    return PaddingCoordinates(top=0, bottom=pad_height, left=0, right=pad_width)


def _pad_image(image: np.ndarray, padding_coordinates: PaddingCoordinates, pad_value: int) -> np.ndarray:
    """Pad an image.

    :param image:       Image to shift. (H, W, C) or (H, W).
    :param pad_h:       Tuple of (padding_top, padding_bottom).
    :param pad_w:       Tuple of (padding_left, padding_right).
    :param pad_value:   Padding value
    :return:            Image shifted according to padding coordinates.
    """
    pad_h = (padding_coordinates.top, padding_coordinates.bottom)
    pad_w = (padding_coordinates.left, padding_coordinates.right)

    if len(image.shape) == 3:
        return np.pad(image, (pad_h, pad_w, (0, 0)), "constant", constant_values=pad_value)
    else:
        return np.pad(image, (pad_h, pad_w), "constant", constant_values=pad_value)


def _shift_bboxes(targets: np.array, shift_w: float, shift_h: float) -> np.array:
    """Shift bboxes with respect to padding values.

    :param targets:  Bboxes to transform of shape (N, 4+), in format [x1, y1, x2, y2, ...]
    :param shift_w:  shift width.
    :param shift_h:  shift height.
    :return:         Bboxes transformed of shape (N, 4+), in format [x1, y1, x2, y2, ...]
    """
    boxes, labels = targets[:, :4], targets[:, 4:]
    boxes[:, [0, 2]] += shift_w
    boxes[:, [1, 3]] += shift_h
    return np.concatenate((boxes, labels), 1)


def _shift_keypoints(targets: np.array, shift_w: float, shift_h: float) -> np.array:
    """Shift keypoints with respect to padding values.

    :param targets:  Keypoints to transform of shape (N, 2+), or (N, K, 2+), in format [x1, y1, ...]
    :param shift_w:  shift width.
    :param shift_h:  shift height.
    :return:         Transformed keypoints of the same shape as input.
    """
    targets = targets.copy()
    targets[..., 0] += shift_w
    targets[..., 1] += shift_h
    return targets


def _rescale_xyxy_bboxes(targets: np.array, r: float) -> np.array:
    """Scale targets to given scale factors.

    :param targets:  Bboxes to transform of shape (N, 4+), in format [x1, y1, x2, y2, ...]
    :param r:        DetectionRescale coefficient that was applied to the image
    :return:         Rescaled Bboxes to transform of shape (N, 4+), in format [x1, y1, x2, y2, ...]
    """
    targets = targets.copy()
    boxes, targets = targets[:, :4], targets[:, 4:]
    boxes = xyxy2cxcywh(boxes)
    boxes *= r
    boxes = cxcywh2xyxy(boxes)
    return np.concatenate((boxes, targets), 1)


def _rescale_and_pad_to_size(image: np.ndarray, output_shape: Tuple[int, int], swap: Tuple[int] = (2, 0, 1), pad_val: int = 114) -> Tuple[np.ndarray, float]:
    """
    Rescales image according to minimum ratio input height/width and output height/width rescaled_padded_image,
    pads the image to the target shape and finally swap axis.
    Note: Pads the image to corner, padding is not centered.

    :param image:           Image to be rescaled. (H, W, C) or (H, W).
    :param output_shape:    Target Shape.
    :param swap:            Axis's to be rearranged.
    :param pad_val:         Value to use for padding.
    :return:
        - Rescaled image while preserving aspect ratio, padded to fit output_shape and with axis swapped. By default, (C, H, W).
        - Minimum ratio between the input height/width and output height/width.
    """
    r = min(output_shape[0] / image.shape[0], output_shape[1] / image.shape[1])
    rescale_shape = (int(image.shape[0] * r), int(image.shape[1] * r))

    resized_image = _rescale_image(image=image, target_shape=rescale_shape)

    padding_coordinates = _get_bottom_right_padding_coordinates(input_shape=rescale_shape, output_shape=output_shape)
    padded_image = _pad_image(image=resized_image, padding_coordinates=padding_coordinates, pad_value=pad_val)

    padded_image = padded_image.transpose(swap)
    padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
    return padded_image, r
