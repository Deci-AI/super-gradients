from typing import Tuple

import cv2
import numpy as np

from super_gradients.training.utils.detection_utils import xyxy2cxcywh, cxcywh2xyxy


def _rescale_image(image: np.ndarray, target_shape: Tuple[float, float]) -> np.ndarray:
    """Rescale image to target_shape, without preserving aspect ratio.

    :param image:           Image to rescale. (H, W, C) or (H, W).
    :param target_shape:    Target shape to rescale to.
    :return:                Rescaled image.
    """
    return cv2.resize(image, dsize=(int(target_shape[1]), int(target_shape[0])), interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def _rescale_bboxes(targets: np.array, scale_factors: Tuple[float, float]) -> np.array:
    """Rescale bboxes to given scale factors, without preserving aspect ratio.

    :param targets:         Targets to rescale (N, 4+), where target[:, :4] is the bounding box coordinates.
    :param scale_factors:   Tuple of (sy, sx) scale factors to rescale to.
    :return:                Rescaled targets.
    """

    targets = targets.astype(np.float32, copy=True)

    sy, sx = scale_factors
    targets[:, :4] *= np.array([[sx, sy, sx, sy]], dtype=targets.dtype)
    return targets


def _get_center_padding_params(input_size: Tuple[int, int], output_size: Tuple[int, int]) -> Tuple[int, int, Tuple[int, int], Tuple[int, int]]:
    """Get parameters for padding an image to given output size, in center mode.

    :param input_size:  Size of the input image.
    :param output_size: Size to resize to.
    :return:
        - shift_h:  Horizontal shift.
        - shift_w:  Vertical shift.
        - pad_h:    Horizontal padding.
        - pad_w:    Vertical padding.
    """
    pad_h, pad_w = output_size[0] - input_size[0], output_size[1] - input_size[1]
    shift_h, shift_w = pad_h // 2, pad_w // 2
    pad_h = (shift_h, pad_h - shift_h)
    pad_w = (shift_w, pad_w - shift_w)
    return shift_h, shift_w, pad_h, pad_w


def _shift_image(image: np.ndarray, pad_h: Tuple[int, int], pad_w: Tuple[int, int], pad_value: int) -> np.ndarray:
    """Shift bboxes with respect to padding coordinates.

    :param image:       Image to shift. (H, W, C) or (H, W).
    :param pad_h:       Padding to add to height
    :param pad_w:       Padding to add to width
    :param pad_value:   Padding value
    :return:            Image shifted according to padding coordinates.
    """
    return np.pad(image, (pad_h, pad_w, (0, 0)), "constant", constant_values=pad_value)


def _shift_bboxes(targets: np.array, shift_w: float, shift_h: float) -> np.array:
    """Shift bboxes with respect to padding values.

    :param targets:  Bboxes to transform of shape (N, 4+), in format [x1, y1, x2, y2, ..., ...]
    :param shift_w:  shift width.
    :param shift_h:  shift height.
    :return:         Bboxes transformed of shape (N, 4+), in format [x1, y1, x2, y2, ..., ...]
    """
    boxes, labels = targets[:, :4], targets[:, 4:]
    boxes[:, [0, 2]] += shift_w
    boxes[:, [1, 3]] += shift_h
    return np.concatenate((boxes, labels), 1)


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


def _rescale_and_pad_to_size(image: np.ndarray, output_size: Tuple[int, int], swap: Tuple[int] = (2, 0, 1), pad_val: int = 114) -> Tuple[np.ndarray, float]:
    """
    Rescales image according to minimum ratio input height/width and output height/width rescaled_padded_image,
    pads the image to the target size and finally swap axis.
    Note: Pads the image to corner, padding is not centered.

    :param image:       Image to be rescaled. (H, W, C) or (H, W).
    :param output_size: Target size.
    :param swap:        Axis's to be rearranged.
    :param pad_val:     Value to use for padding.
    :return:
        - Rescaled image while preserving aspect ratio, padded to fit output_size and with axis swapped. By default, (C, H, W).
        - Minimum ratio between the input height/width and output height/width.
    """
    r = min(output_size[0] / image.shape[0], output_size[1] / image.shape[1])
    rescale_shape = (int(image.shape[0] * r), int(image.shape[1] * r))

    resized_image = _rescale_image(image=image, target_shape=rescale_shape)
    padded_image = _pad_image_on_side(image=resized_image, output_size=output_size, pad_val=pad_val)

    padded_image = padded_image.transpose(swap)
    padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
    return padded_image, r


def _pad_image_on_side(image: np.ndarray, output_size: Tuple[int, int], pad_val: int = 114) -> np.ndarray:
    """Pads an image to the specified output size by adding padding only on the sides.

    :param image:       Input image to pad. (H, W, C) or (H, W).
    :param output_size: Expected size of the output image (height, width).
    :param pad_val:     Value to use for padding.
    :return:            Padded image of size output_size.
    """
    if len(image.shape) == 3:
        padded_image = np.ones((output_size[0], output_size[1], image.shape[-1]), dtype=np.uint8) * pad_val
    else:
        padded_image = np.ones(output_size, dtype=np.uint8) * pad_val

    padded_image[: image.shape[0], : image.shape[1]] = image
    return padded_image
