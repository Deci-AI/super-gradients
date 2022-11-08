from abc import abstractmethod
from typing import Tuple

import torch
from torch import Tensor

__all__ = [
    "XYXYCoordinateFormat",
    "XYWHCoordinateFormat",
    "YXYXCoordinateFormat",
    "CXCYWHCoordinateFormat",
    "NormalizedXYWHCoordinateFormat",
    "NormalizedXYXYCoordinateFormat",
    "NormalizedCXCYWHCoordinateFormat",
    "cxcywh2xyxy",
    "normalizedxyxy2xyxy",
    "xywh2xyxy",
    "xyxy2cxcywh",
    "xyxy2normalizedxywh",
    "xyxy2normalizedxyxy",
    "xyxy2xywh",
    "convert_bboxes",
]


def normalizedxyxy2xyxy(bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    """
    Convert unit-normalized XYXY bboxes to XYXY bboxes in pixel units.
    :param bboxes: BBoxes of shape (..., 4) in XYXY (unit-normalized) format
    :param image_shape: Image shape (rows,cols)
    :return: BBoxes of shape (..., 4) in XYXY (pixels) format
    """
    rows, cols = image_shape
    scale = torch.tensor([cols, rows, cols, rows], dtype=bboxes.dtype, device=bboxes.device)
    scale = scale.reshape([1] * (len(bboxes.size()) - 1) + [4])
    return bboxes * scale


def xyxy2normalizedxyxy(bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    """
    Convert bboxes from XYXY (pixels) format to XYXY (unit-normalized) format
    :param bboxes: BBoxes of shape (..., 4) in XYXY (pixels) format
    :param image_shape: Image shape (rows,cols)
    :return: BBoxes of shape (..., 4) in XYXY (unit-normalized) format
    """
    rows, cols = image_shape
    scale = torch.tensor([cols, rows, cols, rows], dtype=bboxes.dtype, device=bboxes.device)
    scale = scale.reshape([1] * (len(bboxes.size()) - 1) + [4])
    return bboxes / scale


def xyxy2xywh(bboxes: Tensor) -> Tensor:
    """
    Transforms bboxes inplace from XYXY format to XYWH format
    :param bboxes: BBoxes of shape (..., 4) in XYXY format
    :return: BBoxes of shape (..., 4) in XYWH format
    """
    x1, y1, x2, y2 = torch.split(bboxes, split_size_or_sections=1, dim=-1)
    w = x2 - x1
    h = y2 - y1
    return torch.cat([x1, y1, w, h], dim=-1)


def xyxy2cxcywh(bboxes: Tensor):
    """
    Transforms bboxes from xyxy format to CX-CY-W-H format
    :param bboxes: BBoxes of shape (..., 4) in XYXY format
    :return: BBoxes of shape (..., 4) in CX-CY-W-H format
    """
    x1, y1, x2, y2 = torch.split(bboxes, split_size_or_sections=1, dim=-1)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.cat([cx, cy, w, h], dim=-1)


def cxcywh2xyxy(bboxes: Tensor) -> Tensor:
    """
    Transforms bboxes from CX-CY-W-H format to XYXY format
    :param bboxes: BBoxes of shape (..., 4) in CX-CY-W-H format
    :return: BBoxes of shape (..., 4) in XYXY format
    """
    cx, cy, w, h = torch.split(bboxes, split_size_or_sections=1, dim=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = x1 + w
    y2 = y1 + h
    return torch.cat([x1, y1, x2, y2], dim=-1)


def xywh2xyxy(bboxes: Tensor):
    """
    Transforms bboxes from XYWH format to XYXY format
    :param bboxes: BBoxes of shape (..., 4) in XYWH format
    :return: BBoxes of shape (..., 4) in XYXY format
    """
    x1, y1, w, h = torch.split(bboxes, split_size_or_sections=1, dim=-1)
    x2 = x1 + w
    y2 = y1 + h
    return torch.cat([x1, y1, x2, y2], dim=-1)


def xyxy2normalizedxywh(bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    """
    Transforms bboxes inplace from xyxy format to unit-normalized xywh format
    :param bboxes: array, shaped (nboxes, 4)
    :return: modified bboxes
    """
    bboxes = xyxy2normalizedxyxy(bboxes, image_shape)
    return xyxy2xywh(bboxes)


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


class NormalizedCXCYWHCoordinateFormat(BoundingBoxFormat):
    def __init__(self):
        self.format = "cxcywh"
        self.normalized = True

    def to_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        return normalizedxyxy2xyxy(cxcywh2xyxy(bboxes), image_shape)

    def from_xyxy(self, bboxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
        return xyxy2cxcywh(xyxy2normalizedxyxy(bboxes, image_shape))


def convert_bboxes(bboxes: Tensor, image_shape: Tuple[int, int], source_format: BoundingBoxFormat, target_format: BoundingBoxFormat):
    """
    Convert bboxes from source to target format
    :param bboxes: Tensor of shape (..., 4) with input bounding boxes
    :param image_shape: Tuple of (rows, cols) corresponding to image shape
    :param source_format: Format of the source bounding boxes
    :param target_format: Format of the output bounding boxes
    :return: Tensor of shape (..., 4) with resulting bounding boxes
    """
    return target_format.from_xyxy(source_format.to_xyxy(bboxes, image_shape), image_shape)
