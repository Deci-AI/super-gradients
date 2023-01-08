import warnings
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from super_gradients.training.datasets.data_formats.bbox_formats.bbox_format import (
    BoundingBoxFormat,
)

__all__ = [
    "NormalizedXYXYCoordinateFormat",
    "normalized_xyxy_to_xyxy",
    "normalized_xyxy_to_xyxy_inplace",
    "xyxy_to_normalized_xyxy",
    "xyxy_to_normalized_xyxy_inplace",
]


def normalized_xyxy_to_xyxy(bboxes, image_shape: Tuple[int, int]):
    """
    Convert unit-normalized XYXY bboxes to XYXY bboxes in pixel units.
    :param bboxes: BBoxes of shape (..., 4) in XYXY (unit-normalized) format
    :param image_shape: Image shape (rows,cols)
    :return: BBoxes of shape (..., 4) in XYXY (pixels) format
    """
    rows, cols = image_shape
    if torch.jit.is_scripting():
        scale = torch.tensor([cols, rows, cols, rows], dtype=bboxes.dtype, device=bboxes.device)
        scale = scale.reshape([1] * (len(bboxes.size()) - 1) + [4])
    else:
        if torch.is_tensor(bboxes):
            scale = torch.tensor([cols, rows, cols, rows], dtype=bboxes.dtype, device=bboxes.device)
            scale = scale.reshape([1] * (len(bboxes.size()) - 1) + [4])
        elif isinstance(bboxes, np.ndarray):
            scale = np.array([cols, rows, cols, rows], dtype=bboxes.dtype)
            scale = scale.reshape([1] * (len(bboxes.shape) - 1) + [4])
        else:
            raise RuntimeError(f"Only Torch tensor or Numpy array is supported. Received bboxes of type {str(type(bboxes))}")

    return bboxes * scale


def xyxy_to_normalized_xyxy(bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    """
    Convert bboxes from XYXY (pixels) format to XYXY (unit-normalized) format
    :param bboxes: BBoxes of shape (..., 4) in XYXY (pixels) format
    :param image_shape: Image shape (rows,cols)
    :return: BBoxes of shape (..., 4) in XYXY (unit-normalized) format
    """
    rows, cols = image_shape
    if torch.jit.is_scripting():
        scale = torch.tensor([cols, rows, cols, rows], dtype=bboxes.dtype, device=bboxes.device)
        scale = scale.reshape([1] * (len(bboxes.size()) - 1) + [4])
    else:
        if torch.is_tensor(bboxes):
            scale = torch.tensor([cols, rows, cols, rows], dtype=bboxes.dtype, device=bboxes.device)
            scale = scale.reshape([1] * (len(bboxes.size()) - 1) + [4])
        elif isinstance(bboxes, np.ndarray):
            scale = np.array([cols, rows, cols, rows], dtype=bboxes.dtype)
        else:
            raise RuntimeError(f"Only Torch tensor or Numpy array is supported. Received bboxes of type {str(type(bboxes))}")
    return bboxes / scale


def normalized_xyxy_to_xyxy_inplace(bboxes, image_shape: Tuple[int, int]):
    """
    Convert unit-normalized XYXY bboxes to XYXY bboxes in pixel units. This function operates in-place.
    :param bboxes: BBoxes of shape (..., 4) in XYXY (unit-normalized) format
    :param image_shape: Image shape (rows,cols)
    :return: BBoxes of shape (..., 4) in XYXY (pixels) format
    """
    rows, cols = image_shape
    bboxes[..., 0:3:2] *= cols
    bboxes[..., 1:4:2] *= rows
    return bboxes


def xyxy_to_normalized_xyxy_inplace(bboxes, image_shape: Tuple[int, int]):
    """
    Convert bboxes from XYXY (pixels) format to XYXY (unit-normalized) format. This function operates in-place.
    :param bboxes: BBoxes of shape (..., 4) in XYXY (pixels) format
    :param image_shape: Image shape (rows,cols)
    :return: BBoxes of shape (..., 4) in XYXY (unit-normalized) format
    """

    if not torch.jit.is_scripting():
        if torch.is_tensor(bboxes) and not torch.is_floating_point(bboxes):
            warnings.warn(
                f"Detected non floating-point ({bboxes.dtype}) input to xyxy_to_normalized_xyxy_inplace function. "
                f"This may cause rounding errors and lose of precision. You may want to convert your array to floating-point precision first."
            )
        if isinstance(bboxes, np.ndarray) and not np.issubdtype(bboxes.dtype, np.floating):
            warnings.warn(
                f"Detected non floating-point input ({bboxes.dtype}) to xyxy_to_normalized_xyxy_inplace function. "
                f"This may cause rounding errors and lose of precision. You may want to convert your array to floating-point precision first."
            )

    rows, cols = image_shape
    bboxes[..., 0:3:2] /= cols
    bboxes[..., 1:4:2] /= rows
    return bboxes


class NormalizedXYXYCoordinateFormat(BoundingBoxFormat):
    """
    Normalized X1,Y1,X2,Y2 bounding boxes format
    """

    def __init__(self):
        super().__init__()
        self.format = "normalized_xyxy"
        self.normalized = True

    def get_to_xyxy(self, inplace: bool):
        if inplace:
            return normalized_xyxy_to_xyxy_inplace
        else:
            return normalized_xyxy_to_xyxy

    def get_from_xyxy(self, inplace: bool):
        if inplace:
            return xyxy_to_normalized_xyxy_inplace
        else:
            return xyxy_to_normalized_xyxy
