from typing import Tuple

from super_gradients.training.datasets.data_formats.bbox_formats.bbox_format import (
    BoundingBoxFormat,
)
from super_gradients.training.datasets.data_formats.bbox_formats.normalized_xyxy import (
    normalized_xyxy_to_xyxy_inplace,
    xyxy_to_normalized_xyxy_inplace,
    xyxy_to_normalized_xyxy,
)
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy_inplace, xywh_to_xyxy, xyxy_to_xywh_inplace

__all__ = [
    "xyxy_to_normalized_xywh",
    "normalized_xywh_to_xyxy_inplace",
    "xyxy_to_normalized_xywh_inplace",
    "normalized_xywh_to_xyxy",
    "NormalizedXYWHCoordinateFormat",
]


def normalized_xywh_to_xyxy(bboxes, image_shape: Tuple[int, int]):
    normalized_xyxy = xywh_to_xyxy(bboxes, image_shape)  # Out-of-place
    return normalized_xyxy_to_xyxy_inplace(normalized_xyxy, image_shape)  # Can be done inplace


def normalized_xywh_to_xyxy_inplace(bboxes, image_shape: Tuple[int, int]):
    normalized_xyxy = xywh_to_xyxy_inplace(bboxes, image_shape)
    return normalized_xyxy_to_xyxy_inplace(normalized_xyxy, image_shape)


def xyxy_to_normalized_xywh(bboxes, image_shape: Tuple[int, int]):
    normalized_xyxy = xyxy_to_normalized_xyxy(bboxes, image_shape)
    return xyxy_to_xywh_inplace(normalized_xyxy, image_shape)  # Can be done inplace


def xyxy_to_normalized_xywh_inplace(bboxes, image_shape: Tuple[int, int]):
    normalized_xyxy = xyxy_to_normalized_xyxy_inplace(bboxes, image_shape)
    return xyxy_to_xywh_inplace(normalized_xyxy, image_shape)


class NormalizedXYWHCoordinateFormat(BoundingBoxFormat):
    def __init__(self):
        self.format = "normalized_xywh"
        self.normalized = True

    def get_to_xyxy(self, inplace: bool):
        if inplace:
            return normalized_xywh_to_xyxy_inplace
        else:
            return normalized_xywh_to_xyxy

    def get_from_xyxy(self, inplace: bool):
        if inplace:
            return xyxy_to_normalized_xywh_inplace
        else:
            return xyxy_to_normalized_xywh
