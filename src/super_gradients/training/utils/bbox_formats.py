from abc import abstractmethod
from typing import Tuple

import torch
from torch import Tensor
from .detection_utils import xyxy2xywh, xyxy2cxcywh, xyxy2normalizedxywh, xyxy2normalizedxyxy, normalizedxyxy2xyxy, cxcywh2xyxy

__all__ = [
    "XYXYCoordinateFormat",
    "YXYXCoordinateFormat",
    "XYWHCoordinateFormat",
    "CXCYWHCoordinateFormat",
    "NormalizedXYWHCoordinateFormat",
    "NormalizedXYXYCoordinateFormat",
]


def normalizedxyxy2xyxy(bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    scale = torch.tensor([[image_shape[1], image_shape[0], image_shape[1], image_shape[0]]], dtype=bboxes.dtype, device=bboxes.device)
    return bboxes * scale


def xyxy2xywh(bboxes: Tensor):
    """
    Transforms bboxes inplace from xyxy format to xywh format
    :param bboxes: array, shaped (nboxes, 4)
    :return: modified bboxes
    """
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2normalizedxyxy(bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    """
    Transforms bboxes inplace from xyxy format to unit-normalized xyxy format
    :param bboxes: array, shaped (nboxes, 4)
    :param image_shape: Image shape (rows,cols)
    :return: modified bboxes
    """

    scale = torch.reciprocal(torch.tensor([[image_shape[1], image_shape[0], image_shape[1], image_shape[0]]], dtype=bboxes.dtype, device=bboxes.device))
    return bboxes * scale


def xyxy2normalizedxywh(bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    """
    Transforms bboxes inplace from xyxy format to unit-normalized xywh format
    :param bboxes: array, shaped (nboxes, 4)
    :return: modified bboxes
    """
    bboxes = xyxy2normalizedxyxy(bboxes, image_shape)
    return xyxy2xywh(bboxes)


def xyxy2normalizedcxcywh(bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    """
    Transforms bboxes inplace from centerized xy wh format to xyxy format
    :param bboxes: array, shaped (nboxes, 4)
    :return: modified bboxes
    """
    bboxes = xyxy2normalizedxyxy(bboxes, image_shape)
    return xyxy2cxcywh(bboxes)


class BoundingBoxFormat:
    """
    Abstract class for describing a bounding boxes format. It exposes two methods: to_xyxy and from_xyxy to convert
    whatever format of boxes we are dealing with to internal xyxy format and vice versa. This conversion from and to
    intermediate xyxy format has a subtle performance impact, but greatly reduce amount of boilerplate code to support
    all combinations of conversion xyxy, xywh, cxcywh, yxyx <-> xyxy, xywh, cxcywh, yxyx.
    """

    @abstractmethod
    def to_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        """
        Convert input boxes to XYXY format
        :param bboxes: Input bounding boxes [..., 4]
        :param image_shape: Dimensions (rows, cols) of the original image to support
                            normalized boxes or non top-left origin coordinate system.
        :return: Converted bounding boxes [..., 4] in XYXY format
        """
        raise NotImplementedError()

    @abstractmethod
    def from_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        """
        Convert XYXY boxes to target bboxes format
        :param bboxes: Input bounding boxes [..., 4] in XYXY format
        :param image_shape: Dimensions (rows, cols) of the original image to support
                            normalized boxes or non top-left origin coordinate system.
        :return: Converted bounding boxes [..., 4] in target format
        """
        raise NotImplementedError()


class XYXYCoordinateFormat(BoundingBoxFormat):
    """
    Bounding boxes format X1, Y1, X2, Y2
    """

    def __init__(self):
        self.format = "xyxy"
        self.normalized = False

    def to_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        return bboxes

    def from_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        return bboxes


class YXYXCoordinateFormat(BoundingBoxFormat):
    """
    Bounding boxes format Y1, X1, Y2, X1
    """

    def __init__(self):
        super().__init__()
        self.format = "yxyx"
        self.normalized = False

    def to_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        return bboxes[..., torch.tensor([1, 0, 3, 2], dtype=torch.long, device=bboxes.device)]

    def from_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        return bboxes[..., torch.tensor([1, 0, 3, 2], dtype=torch.long, device=bboxes.device)]


class XYWHCoordinateFormat(BoundingBoxFormat):
    def __init__(self):
        self.format = "xywh"
        self.normalized = False

    def to_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        return xywh2xyxy(bboxes)

    def from_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        return xyxy2xywh(bboxes)


class CXCYWHCoordinateFormat(BoundingBoxFormat):
    def __init__(self):
        self.format = "cxcywh"
        self.normalized = False

    def to_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        return cxcywh2xyxy(bboxes)

    def from_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        return xyxy2cxcywh(bboxes)


class NormalizedXYWHCoordinateFormat(BoundingBoxFormat):
    def __init__(self):
        self.format = "normalized_xywh"
        self.normalized = True

    def to_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        return normalizedxyxy2xyxy(xywh2xyxy(bboxes), image_shape)

    def from_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        return xyxy2normalizedxywh(bboxes, image_shape)


class NormalizedXYXYCoordinateFormat(BoundingBoxFormat):
    """
    Normalized X1,Y1,X2,Y2 bounding boxes format
    """

    def __init__(self):
        super().__init__()
        self.format = "normalized_xyxy"
        self.normalized = True

    def to_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        return normalizedxyxy2xyxy(bboxes, image_shape)

    def from_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        return xyxy2normalizedxyxy(bboxes, image_shape)
