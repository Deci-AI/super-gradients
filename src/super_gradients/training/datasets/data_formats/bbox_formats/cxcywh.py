import warnings
from typing import Any, Tuple, Union

import numpy as np
import torch

from super_gradients.training.datasets.data_formats.bbox_formats.bbox_format import (
    BoundingBoxFormat,
)

__all__ = ["xyxy_to_cxcywh", "xyxy_to_cxcywh_inplace", "cxcywh_to_xyxy_inplace", "cxcywh_to_xyxy", "CXCYWHCoordinateFormat"]


def xyxy_to_cxcywh(bboxes, image_shape: Tuple[int, int]):
    """
    Transforms bboxes from xyxy format to CX-CY-W-H format
    :param bboxes: BBoxes of shape (..., 4) in XYXY format
    :return: BBoxes of shape (..., 4) in CX-CY-W-H format
    """
    x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    w = x2 - x1
    h = y2 - y1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    if torch.jit.is_scripting():
        return torch.stack([cx, cy, w, h], dim=-1)
    else:
        if torch.is_tensor(bboxes):
            return torch.stack([cx, cy, w, h], dim=-1)
        elif isinstance(bboxes, np.ndarray):
            return np.stack([cx, cy, w, h], axis=-1)
        else:
            raise RuntimeError(f"Only Torch tensor or Numpy array is supported. Received bboxes of type {str(type(bboxes))}")


def cxcywh_to_xyxy(bboxes, image_shape: Tuple[int, int]):
    """
    Transforms bboxes from CX-CY-W-H format to XYXY format
    :param bboxes: BBoxes of shape (..., 4) in CX-CY-W-H format
    :return: BBoxes of shape (..., 4) in XYXY format
    """
    cx, cy, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = x1 + w
    y2 = y1 + h

    if torch.jit.is_scripting():
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        if torch.is_tensor(bboxes):
            return torch.stack([x1, y1, x2, y2], dim=-1)
        if isinstance(bboxes, np.ndarray):
            return np.stack([x1, y1, x2, y2], axis=-1)
        else:
            raise RuntimeError(f"Only Torch tensor or Numpy array is supported. Received bboxes of type {str(type(bboxes))}")


def is_floating_point_array(array: Union[np.ndarray, Any]) -> bool:
    return isinstance(array, np.ndarray) and np.issubdtype(array.dtype, np.floating)


def cxcywh_to_xyxy_inplace(bboxes, image_shape: Tuple[int, int]):
    """
    Not that bboxes dtype is preserved, and it may lead to unwanted rounding errors when computing a center of bbox.

    :param bboxes: BBoxes of shape (..., 4) in CX-CY-W-H format
    :return: BBoxes of shape (..., 4) in XYXY format
    """
    if not torch.jit.is_scripting():
        if torch.is_tensor(bboxes) and not torch.is_floating_point(bboxes):
            warnings.warn(
                f"Detected non floating-point ({bboxes.dtype}) input to cxcywh_to_xyxy_inplace function. "
                f"This may cause rounding errors and lose of precision. You may want to convert your array to floating-point precision first."
            )
        if not is_floating_point_array(bboxes):
            warnings.warn(
                f"Detected non floating-point input ({bboxes.dtype}) to cxcywh_to_xyxy_inplace function. "
                f"This may cause rounding errors and lose of precision. You may want to convert your array to floating-point precision first."
            )

    bboxes[..., 0:2] -= bboxes[..., 2:4] * 0.5  # cxcy -> x1y1
    bboxes[..., 2:4] += bboxes[..., 0:2]  # x1y1 + wh -> x2y2
    return bboxes


def xyxy_to_cxcywh_inplace(bboxes, image_shape: Tuple[int, int]):
    """
    Transforms bboxes from xyxy format to CX-CY-W-H format. This function operates in-place.
    Not that bboxes dtype is preserved, and it may lead to unwanted rounding errors when computing a center of bbox.

    :param bboxes: BBoxes of shape (..., 4) in XYXY format
    :return: BBoxes of shape (..., 4) in CX-CY-W-H format
    """
    if not torch.jit.is_scripting():
        if torch.is_tensor(bboxes) and not torch.is_floating_point(bboxes):
            warnings.warn(
                f"Detected non floating-point ({bboxes.dtype}) input to xyxy_to_cxcywh_inplace function. This may cause rounding errors and lose of precision. "
                "You may want to convert your array to floating-point precision first."
            )
        elif isinstance(bboxes, np.ndarray) and not is_floating_point_array(bboxes):
            warnings.warn(
                f"Detected non floating-point input ({bboxes.dtype}) to xyxy_to_cxcywh_inplace function. This may cause rounding errors and lose of precision. "
                "You may want to convert your array to floating-point precision first."
            )
    bboxes[..., 2:4] -= bboxes[..., 0:2]  # x2y2 - x1y2 -> wh
    bboxes[..., 0:2] += bboxes[..., 2:4] * 0.5  # cxcywh
    return bboxes


class CXCYWHCoordinateFormat(BoundingBoxFormat):
    def __init__(self):
        self.format = "cxcywh"
        self.normalized = False

    def get_to_xyxy(self, inplace: bool):
        if inplace:
            return cxcywh_to_xyxy_inplace
        else:
            return cxcywh_to_xyxy

    def get_from_xyxy(self, inplace: bool):
        if inplace:
            return xyxy_to_cxcywh_inplace
        else:
            return xyxy_to_cxcywh
