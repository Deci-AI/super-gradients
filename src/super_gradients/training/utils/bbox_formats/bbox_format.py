from abc import abstractmethod
from typing import Union, Tuple

import numpy as np
import torch
from torch import Tensor

__all__ = ["BoundingBoxFormat", "convert_bboxes", "normalizedxyxy2xyxy_inplace", "normalizedxyxy2xyxy", "xyxy2normalizedxyxy_inplace", "xyxy2normalizedxyxy"]


def normalizedxyxy2xyxy(bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int]) -> Union[Tensor, np.ndarray]:
    """
    Convert unit-normalized XYXY bboxes to XYXY bboxes in pixel units.
    :param bboxes: BBoxes of shape (..., 4) in XYXY (unit-normalized) format
    :param image_shape: Image shape (rows,cols)
    :return: BBoxes of shape (..., 4) in XYXY (pixels) format
    """
    rows, cols = image_shape
    if torch.is_tensor(bboxes):
        scale = torch.tensor([cols, rows, cols, rows], dtype=bboxes.dtype, device=bboxes.device)
        scale = scale.reshape([1] * (len(bboxes.size()) - 1) + [4])
    elif isinstance(bboxes, np.ndarray):
        scale = np.array([cols, rows, cols, rows], dtype=bboxes.dtype)
        scale = scale.reshape([1] * (len(bboxes.shape) - 1) + [4])
    else:
        raise RuntimeError(f"Only Torch tensor or Numpy array is supported. Received bboxes of type {str(type(bboxes))}")

    return bboxes * scale


def xyxy2normalizedxyxy(bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    """
    Convert bboxes from XYXY (pixels) format to XYXY (unit-normalized) format
    :param bboxes: BBoxes of shape (..., 4) in XYXY (pixels) format
    :param image_shape: Image shape (rows,cols)
    :return: BBoxes of shape (..., 4) in XYXY (unit-normalized) format
    """
    rows, cols = image_shape
    if torch.is_tensor(bboxes):
        scale = torch.tensor([cols, rows, cols, rows], dtype=bboxes.dtype, device=bboxes.device)
        scale = scale.reshape([1] * (len(bboxes.size()) - 1) + [4])
    elif isinstance(bboxes, np.ndarray):
        scale = np.array([cols, rows, cols, rows], dtype=bboxes.dtype)
    else:
        raise RuntimeError(f"Only Torch tensor or Numpy array is supported. Received bboxes of type {str(type(bboxes))}")
    return bboxes / scale


def normalizedxyxy2xyxy_inplace(bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int]) -> Union[Tensor, np.ndarray]:
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


def xyxy2normalizedxyxy_inplace(bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    """
    Convert bboxes from XYXY (pixels) format to XYXY (unit-normalized) format. This function operates in-place.
    :param bboxes: BBoxes of shape (..., 4) in XYXY (pixels) format
    :param image_shape: Image shape (rows,cols)
    :return: BBoxes of shape (..., 4) in XYXY (unit-normalized) format
    """
    rows, cols = image_shape
    bboxes[..., 0:3:2] /= cols
    bboxes[..., 1:4:2] /= rows
    return bboxes


class BoundingBoxFormat:
    """
    Abstract class for describing a bounding boxes format. It exposes two methods: to_xyxy and from_xyxy to convert
    whatever format of boxes we are dealing with to internal xyxy format and vice versa. This conversion from and to
    intermediate xyxy format has a subtle performance impact, but greatly reduce amount of boilerplate code to support
    all combinations of conversion xyxy, xywh, cxcywh, yxyx <-> xyxy, xywh, cxcywh, yxyx.
    """

    @abstractmethod
    def to_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        """
        Convert input boxes to XYXY format
        :param bboxes: Input bounding boxes [..., 4]
        :param image_shape: Dimensions (rows, cols) of the original image to support
                            normalized boxes or non top-left origin coordinate system.
        :return: Converted bounding boxes [..., 4] in XYXY format
        """
        raise NotImplementedError()

    @abstractmethod
    def from_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        """
        Convert XYXY boxes to target bboxes format
        :param bboxes: Input bounding boxes [..., 4] in XYXY format
        :param image_shape: Dimensions (rows, cols) of the original image to support
                            normalized boxes or non top-left origin coordinate system.
        :return: Converted bounding boxes [..., 4] in target format
        """
        raise NotImplementedError()


def convert_bboxes(
    bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], source_format: BoundingBoxFormat, target_format: BoundingBoxFormat, inplace: bool
):
    """
    Convert bboxes from source to target format
    :param bboxes: Tensor of shape (..., 4) with input bounding boxes
    :param image_shape: Tuple of (rows, cols) corresponding to image shape
    :param source_format: Format of the source bounding boxes
    :param target_format: Format of the output bounding boxes
    :return: Tensor of shape (..., 4) with resulting bounding boxes
    """
    xyxy = source_format.to_xyxy(bboxes, image_shape, inplace)
    return target_format.from_xyxy(xyxy, image_shape, inplace)
