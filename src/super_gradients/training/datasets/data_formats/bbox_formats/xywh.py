from typing import Tuple

import numpy as np
import torch

from super_gradients.training.datasets.data_formats.bbox_formats.bbox_format import (
    BoundingBoxFormat,
)

__all__ = ["xyxy_to_xywh", "xywh_to_xyxy_inplace", "xyxy_to_xywh_inplace", "xywh_to_xyxy", "XYWHCoordinateFormat"]


def xyxy_to_xywh(bboxes, image_shape: Tuple[int, int]):
    """
    Transforms bboxes inplace from XYXY format to XYWH format
    :param bboxes: BBoxes of shape (..., 4) in XYXY format
    :return: BBoxes of shape (..., 4) in XYWH format
    """
    x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    w = x2 - x1
    h = y2 - y1

    if torch.jit.is_scripting():
        return torch.stack([x1, y1, w, h], dim=-1)
    else:
        if torch.is_tensor(bboxes):
            return torch.stack([x1, y1, w, h], dim=-1)
        elif isinstance(bboxes, np.ndarray):
            return np.stack([x1, y1, w, h], axis=-1)
        else:
            raise RuntimeError(f"Only Torch tensor or Numpy array is supported. Received bboxes of type {str(type(bboxes))}")


def xywh_to_xyxy(bboxes, image_shape: Tuple[int, int]):
    """
    Transforms bboxes from XYWH format to XYXY format
    :param bboxes: BBoxes of shape (..., 4) in XYWH format
    :return: BBoxes of shape (..., 4) in XYXY format
    """
    x1, y1, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    x2 = x1 + w
    y2 = y1 + h

    if torch.jit.is_scripting():
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        if torch.is_tensor(bboxes):
            return torch.stack([x1, y1, x2, y2], dim=-1)
        elif isinstance(bboxes, np.ndarray):
            return np.stack([x1, y1, x2, y2], axis=-1)
        else:
            raise RuntimeError(f"Only Torch tensor or Numpy array is supported. Received bboxes of type {str(type(bboxes))}")


def xyxy_to_xywh_inplace(bboxes, image_shape: Tuple[int, int]):
    """
    Transforms bboxes inplace from XYXY format to XYWH format. This function operates in-place.
    :param bboxes: BBoxes of shape (..., 4) in XYXY format
    :return: BBoxes of shape (..., 4) in XYWH format
    """
    bboxes[..., 2:4] -= bboxes[..., 0:2]
    return bboxes


def xywh_to_xyxy_inplace(bboxes, image_shape: Tuple[int, int]):
    """
    Transforms bboxes from XYWH format to XYXY format
    :param bboxes: BBoxes of shape (..., 4) in XYWH format
    :return: BBoxes of shape (..., 4) in XYXY format
    """
    bboxes[..., 2:4] += bboxes[..., 0:2]
    return bboxes


class XYWHCoordinateFormat(BoundingBoxFormat):
    def __init__(self):
        self.format = "xywh"
        self.normalized = False

    def get_to_xyxy(self, inplace: bool):
        if inplace:
            return xywh_to_xyxy_inplace
        else:
            return xywh_to_xyxy

    def get_from_xyxy(self, inplace: bool):
        if inplace:
            return xyxy_to_xywh_inplace
        else:
            return xyxy_to_xywh
