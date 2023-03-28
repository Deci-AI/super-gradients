from typing import Union, Tuple, Optional

import cv2
import numpy as np

from PIL import Image
from super_gradients.training.utils.detection_utils import xyxy2cxcywh, cxcywh2xyxy


image_resample = Image.BILINEAR
mask_resample = Image.NEAREST


def _rescale_target(targets: np.array, scale_factors: Tuple[float, float]) -> np.array:
    """DetectionRescale targets to given scale factors."""
    sy, sx = scale_factors
    targets = targets.astype(np.float32, copy=True) if len(targets) > 0 else np.zeros((0, 5), dtype=np.float32)
    targets[:, 0:4] *= np.array([[sx, sy, sx, sy]], dtype=targets.dtype)
    return targets


def _rescale_image(image: np.ndarray, target_shape: Tuple[float, float]) -> np.ndarray:
    """DetectionRescale image to target_shape, without preserving aspect ratio."""
    return cv2.resize(image, dsize=(int(target_shape[1]), int(target_shape[0])), interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def _translate_targets(targets: np.array, shift_w: float, shift_h: float) -> np.array:
    """Translate bboxes with respect to padding values.

    :param targets:  Bboxes to transform of shape (N, 5+), in format [x1, y1, x2, y2, class_id, ...]
    :param shift_w:  shift width in pixels
    :param shift_h:  shift height in pixels
    :return:         Bboxes to transform of shape (N, 5+), in format [x1, y1, x2, y2, class_id, ...]
    """
    targets = targets.copy() if len(targets) > 0 else np.zeros((0, 5), dtype=np.float32)
    boxes, labels = targets[:, :4], targets[:, 4:]
    boxes[:, [0, 2]] += shift_w
    boxes[:, [1, 3]] += shift_h
    return np.concatenate((boxes, labels), 1)


def _rescale_xyxy_target(targets: np.array, r: float) -> np.array:
    """Scale targets to given scale factors.

    :param targets:  Bboxes to transform of shape (N, 5+), in format [x1, y1, x2, y2, class_id, ...]
    :param r:        DetectionRescale coefficient that was applied to the image
    :return:         Rescaled Bboxes to transform of shape (N, 5+), in format [x1, y1, x2, y2, class_id, ...]
    """
    targets = targets.copy()
    boxes, targets = targets[:, :4], targets[:, 4:]
    boxes = xyxy2cxcywh(boxes)
    boxes *= r
    boxes = cxcywh2xyxy(boxes)
    return np.concatenate((boxes, targets), 1)


def _rescale_and_pad_to_size(image: np.ndarray, output_size: Tuple[int, int], swap: Tuple[int] = (2, 0, 1), pad_val: int = 114) -> Tuple[np.ndarray, float]:
    """
    Rescales image according to minimum ratio between the target height /image height, target width / image width,
    and pads the image to the target size.

    :param image:       Image to be rescaled
    :param output_size: Target size
    :param swap:        Axis's to be rearranged.
    :param pad_val:     Value to use for padding
    :return:            Rescaled image according to ratio r and padded to fit output_size.
    """
    if len(image.shape) == 3:
        padded_image = np.ones((output_size[0], output_size[1], image.shape[-1]), dtype=np.uint8) * pad_val
    else:
        padded_image = np.ones(output_size, dtype=np.uint8) * pad_val

    r = min(output_size[0] / image.shape[0], output_size[1] / image.shape[1])

    target_shape = (int(image.shape[0] * r), int(image.shape[1] * r))
    resized_image = _rescale_image(image=image, target_shape=target_shape)
    padded_image[: target_shape[0], : target_shape[1]] = resized_image

    padded_image = padded_image.transpose(swap)
    padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
    return padded_image, r


def segmentation_rescale(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scale_factor: Optional[float] = None,
    short_size: Optional[int] = None,
    long_size: Optional[int] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    w, h = image.size
    if scale_factor is not None:
        scale = scale_factor
    elif short_size is not None:
        scale = short_size / min(w, h)
    else:
        scale = long_size / max(w, h)

    out_size = int(scale * w), int(scale * h)

    image = image.resize(out_size, image_resample)
    if mask is None:
        return image
    mask = mask.resize(out_size, mask_resample)
    return image, mask
