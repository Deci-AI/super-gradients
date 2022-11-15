import torch

from super_gradients.training.utils.bbox_formats.bbox_format import (
    BoundingBoxFormat,
    normalizedxyxy2xyxy_inplace,
    normalizedxyxy2xyxy,
    xyxy2normalizedxyxy,
    xyxy2normalizedxyxy_inplace,
)
from typing import Union, Tuple

import numpy as np
from torch import Tensor


def xyxy2xywh(bboxes: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """
    Transforms bboxes inplace from XYXY format to XYWH format
    :param bboxes: BBoxes of shape (..., 4) in XYXY format
    :return: BBoxes of shape (..., 4) in XYWH format
    """
    x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    w = x2 - x1
    h = y2 - y1
    if torch.is_tensor(bboxes):
        return torch.stack([x1, y1, w, h], dim=-1)
    elif isinstance(bboxes, np.ndarray):
        return np.stack([x1, y1, w, h], axis=-1)
    else:
        raise RuntimeError(f"Only Torch tensor or Numpy array is supported. Received bboxes of type {str(type(bboxes))}")


def xywh2xyxy(bboxes: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """
    Transforms bboxes from XYWH format to XYXY format
    :param bboxes: BBoxes of shape (..., 4) in XYWH format
    :return: BBoxes of shape (..., 4) in XYXY format
    """
    x1, y1, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    x2 = x1 + w
    y2 = y1 + h
    if torch.is_tensor(bboxes):
        return torch.stack([x1, y1, x2, y2], dim=-1)
    elif isinstance(bboxes, np.ndarray):
        return np.stack([x1, y1, x2, y2], axis=-1)
    else:
        raise RuntimeError(f"Only Torch tensor or Numpy array is supported. Received bboxes of type {str(type(bboxes))}")


def xyxy2xywh_inplace(bboxes: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """
    Transforms bboxes inplace from XYXY format to XYWH format. This function operates in-place.
    :param bboxes: BBoxes of shape (..., 4) in XYXY format
    :return: BBoxes of shape (..., 4) in XYWH format
    """
    bboxes[..., 2:3] -= bboxes[..., 0:1]
    return bboxes


def xywh2xyxy_inplace(bboxes: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """
    Transforms bboxes from XYWH format to XYXY format
    :param bboxes: BBoxes of shape (..., 4) in XYWH format
    :return: BBoxes of shape (..., 4) in XYXY format
    """
    bboxes[..., 2:3] += bboxes[..., 0:1]
    return bboxes


class XYWHCoordinateFormat(BoundingBoxFormat):
    def __init__(self):
        self.format = "xywh"
        self.normalized = False

    def to_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        if inplace:
            return xywh2xyxy_inplace(bboxes)
        else:
            return xywh2xyxy(bboxes)

    def from_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        if inplace:
            return xyxy2xywh_inplace(bboxes)
        else:
            return xyxy2xywh(bboxes)


class NormalizedXYWHCoordinateFormat(BoundingBoxFormat):
    def __init__(self):
        self.format = "normalized_xywh"
        self.normalized = True

    def to_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        if inplace:
            return normalizedxyxy2xyxy_inplace(xywh2xyxy_inplace(bboxes), image_shape)
        else:
            return normalizedxyxy2xyxy(xywh2xyxy(bboxes), image_shape)

    def from_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        if inplace:
            return xyxy2normalizedxyxy_inplace(xywh2xyxy_inplace(bboxes), image_shape)
        else:
            return xyxy2xywh(xyxy2normalizedxyxy(bboxes, image_shape))
