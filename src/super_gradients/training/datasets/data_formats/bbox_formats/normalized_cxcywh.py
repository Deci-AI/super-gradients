from typing import Tuple

from super_gradients.training.datasets.data_formats.bbox_formats.bbox_format import (
    BoundingBoxFormat,
)
from super_gradients.training.datasets.data_formats.bbox_formats.cxcywh import cxcywh_to_xyxy, xyxy_to_cxcywh_inplace, cxcywh_to_xyxy_inplace
from super_gradients.training.datasets.data_formats.bbox_formats.normalized_xyxy import (
    xyxy_to_normalized_xyxy_inplace,
    xyxy_to_normalized_xyxy,
    normalized_xyxy_to_xyxy_inplace,
)

__all__ = [
    "NormalizedCXCYWHCoordinateFormat",
    "normalized_cxcywh_to_xyxy",
    "normalized_cxcywh_to_xyxy_inplace",
    "xyxy_to_normalized_cxcywh",
    "xyxy_to_normalized_cxcywh_inplace",
]


def normalized_cxcywh_to_xyxy(bboxes, image_shape: Tuple[int, int]):
    normalized_xyxy = cxcywh_to_xyxy(bboxes, image_shape)  # Out-of-place
    return normalized_xyxy_to_xyxy_inplace(normalized_xyxy, image_shape)  # Can be done inplace


def normalized_cxcywh_to_xyxy_inplace(bboxes, image_shape: Tuple[int, int]):
    normalized_xyxy = cxcywh_to_xyxy_inplace(bboxes, image_shape)
    return normalized_xyxy_to_xyxy_inplace(normalized_xyxy, image_shape)


def xyxy_to_normalized_cxcywh(bboxes, image_shape: Tuple[int, int]):
    normalized_xyxy = xyxy_to_normalized_xyxy(bboxes, image_shape)
    return xyxy_to_cxcywh_inplace(normalized_xyxy, image_shape)  # Can be done inplace


def xyxy_to_normalized_cxcywh_inplace(bboxes, image_shape: Tuple[int, int]):
    normalized_xyxy = xyxy_to_normalized_xyxy_inplace(bboxes, image_shape)
    return xyxy_to_cxcywh_inplace(normalized_xyxy, image_shape)


class NormalizedCXCYWHCoordinateFormat(BoundingBoxFormat):
    def __init__(self):
        self.format = "normalized_cxcywh"
        self.normalized = True

    def get_to_xyxy(self, inplace: bool):
        if inplace:
            return normalized_cxcywh_to_xyxy_inplace
        else:
            return normalized_cxcywh_to_xyxy

    def get_from_xyxy(self, inplace: bool):
        if inplace:
            return xyxy_to_normalized_cxcywh_inplace
        else:
            return xyxy_to_normalized_cxcywh
