from typing import Tuple

from super_gradients.training.datasets.data_formats.bbox_formats.bbox_format import (
    BoundingBoxFormat,
)

__all__ = ["XYXYCoordinateFormat"]


def xyxy_to_xyxy(x, image_shape: Tuple[int, int]):
    return x


class XYXYCoordinateFormat(BoundingBoxFormat):
    """
    Bounding boxes format X1, Y1, X2, Y2
    """

    def __init__(self):
        self.format = "xyxy"
        self.normalized = False

    def get_to_xyxy(self, inplace: bool):
        return xyxy_to_xyxy

    def get_from_xyxy(self, inplace: bool):
        return xyxy_to_xyxy
