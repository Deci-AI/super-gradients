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


class XYXYCoordinateFormat(BoundingBoxFormat):
    """
    Bounding boxes format X1, Y1, X2, Y2
    """

    def __init__(self):
        self.format = "xyxy"
        self.normalized = False

    def to_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        return bboxes

    def from_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        return bboxes


class NormalizedXYXYCoordinateFormat(BoundingBoxFormat):
    """
    Normalized X1,Y1,X2,Y2 bounding boxes format
    """

    def __init__(self):
        super().__init__()
        self.format = "normalized_xyxy"
        self.normalized = True

    def to_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        if inplace:
            return normalizedxyxy2xyxy_inplace(bboxes, image_shape)
        else:
            return normalizedxyxy2xyxy(bboxes, image_shape)

    def from_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        if inplace:
            return xyxy2normalizedxyxy_inplace(bboxes, image_shape)
        else:
            return xyxy2normalizedxyxy(bboxes, image_shape)
