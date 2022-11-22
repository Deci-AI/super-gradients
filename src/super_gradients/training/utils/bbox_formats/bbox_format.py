from abc import abstractmethod
from typing import Tuple, Callable

from torch import Tensor

__all__ = ["BoundingBoxFormat", "convert_bboxes"]


class BoundingBoxFormat:
    """
    Abstract class for describing a bounding boxes format. It exposes two methods: to_xyxy and from_xyxy to convert
    whatever format of boxes we are dealing with to internal xyxy format and vice versa. This conversion from and to
    intermediate xyxy format has a subtle performance impact, but greatly reduce amount of boilerplate code to support
    all combinations of conversion xyxy, xywh, cxcywh, yxyx <-> xyxy, xywh, cxcywh, yxyx.
    """

    def to_xyxy(self, bboxes, image_shape: Tuple[int, int], inplace: bool):
        """
        Convert input boxes to XYXY format
        :param bboxes: Input bounding boxes [..., 4]
        :param image_shape: Dimensions (rows, cols) of the original image to support
                            normalized boxes or non top-left origin coordinate system.
        :return: Converted bounding boxes [..., 4] in XYXY format
        """
        return self.get_to_xyxy(inplace)(bboxes, image_shape)

    def from_xyxy(self, bboxes, image_shape: Tuple[int, int], inplace: bool):
        """
        Convert XYXY boxes to target bboxes format
        :param bboxes: Input bounding boxes [..., 4] in XYXY format
        :param image_shape: Dimensions (rows, cols) of the original image to support
                            normalized boxes or non top-left origin coordinate system.
        :return: Converted bounding boxes [..., 4] in target format
        """
        return self.get_from_xyxy(inplace)(bboxes, image_shape)

    @abstractmethod
    def get_to_xyxy(self, inplace: bool) -> Callable[[Tensor, Tuple[int, int]], Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def get_from_xyxy(self, inplace: bool) -> Callable[[Tensor, Tuple[int, int]], Tensor]:
        raise NotImplementedError()

    def get_num_parameters(self) -> int:
        return 4


def convert_bboxes(bboxes, image_shape: Tuple[int, int], source_format: BoundingBoxFormat, target_format: BoundingBoxFormat, inplace: bool):
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
