import torch

from super_gradients.training.utils.bbox_formats.bbox_format import (
    BoundingBoxFormat,
    normalizedxyxy2xyxy,
    xyxy2normalizedxyxy,
    normalizedxyxy2xyxy_inplace,
    xyxy2normalizedxyxy_inplace,
)
from typing import Union, Tuple

import numpy as np
from torch import Tensor


def xyxy2cxcywh(bboxes: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
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
    if torch.is_tensor(bboxes):
        return torch.stack([cx, cy, w, h], dim=-1)
    elif isinstance(bboxes, np.ndarray):
        return np.stack([cx, cy, w, h], axis=-1)
    else:
        raise RuntimeError(f"Only Torch tensor or Numpy array is supported. Received bboxes of type {str(type(bboxes))}")


def cxcywh2xyxy(bboxes: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
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
    if torch.is_tensor(bboxes):
        return torch.stack([x1, y1, x2, y2], dim=-1)
    elif isinstance(bboxes, np.ndarray):
        return np.stack([x1, y1, x2, y2], axis=-1)
    else:
        raise RuntimeError(f"Only Torch tensor or Numpy array is supported. Received bboxes of type {str(type(bboxes))}")


def cxcywh2xyxy_inplace(bboxes: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """
    Transforms bboxes from CX-CY-W-H format to XYXY format
    :param bboxes: BBoxes of shape (..., 4) in CX-CY-W-H format
    :return: BBoxes of shape (..., 4) in XYXY format
    """
    bboxes[..., 0:1] -= bboxes[..., 2:3] * 0.5  # xywh
    bboxes[..., 2:3] = bboxes[..., 2:3] + bboxes[..., 0:1]  # xyxy
    return bboxes


def xyxy2cxcywh_inplace(bboxes: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """
    Transforms bboxes from xyxy format to CX-CY-W-H format. This function operates in-place.
    :param bboxes: BBoxes of shape (..., 4) in XYXY format
    :return: BBoxes of shape (..., 4) in CX-CY-W-H format
    """
    bboxes[..., 2:3] = bboxes[..., 2:3] - bboxes[..., 0:1]  # xywh
    bboxes[..., 0:1] += bboxes[..., 2:3] * 0.5  # cxcywh
    return bboxes


class CXCYWHCoordinateFormat(BoundingBoxFormat):
    def __init__(self):
        self.format = "cxcywh"
        self.normalized = False

    def to_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        if inplace:
            return cxcywh2xyxy_inplace(bboxes)
        else:
            return cxcywh2xyxy(bboxes)

    def from_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        if inplace:
            return xyxy2cxcywh_inplace(bboxes)
        else:
            return xyxy2cxcywh(bboxes)


class NormalizedCXCYWHCoordinateFormat(BoundingBoxFormat):
    def __init__(self):
        self.format = "normalized_cxcywh"
        self.normalized = True

    def to_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        if inplace:
            return normalizedxyxy2xyxy_inplace(cxcywh2xyxy_inplace(bboxes), image_shape)
        else:
            return normalizedxyxy2xyxy(cxcywh2xyxy(bboxes), image_shape)

    def from_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        if inplace:
            return xyxy2cxcywh_inplace(xyxy2normalizedxyxy_inplace(bboxes, image_shape))
        else:
            return xyxy2cxcywh(xyxy2normalizedxyxy(bboxes, image_shape))
