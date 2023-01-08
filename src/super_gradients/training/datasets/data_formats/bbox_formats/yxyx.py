from typing import Tuple

import numpy as np
import torch

from super_gradients.training.datasets.data_formats.bbox_formats.bbox_format import BoundingBoxFormat

__all__ = ["YXYXCoordinateFormat", "xyxy_to_yxyx", "xyxy_to_yxyx_inplace"]


def xyxy_to_yxyx(bboxes, image_shape: Tuple[int, int]):
    if torch.jit.is_scripting():
        bboxes = torch.moveaxis(bboxes, -1, 0)
        bboxes = bboxes[torch.tensor([1, 0, 3, 2], dtype=torch.long, device=bboxes.device)]
        bboxes = torch.moveaxis(bboxes, 0, 1)
        return bboxes
    else:
        if torch.is_tensor(bboxes):
            return bboxes[..., torch.tensor([1, 0, 3, 2], dtype=torch.long, device=bboxes.device)]
        elif isinstance(bboxes, np.ndarray):
            return bboxes[..., np.array([1, 0, 3, 2], dtype=int)]
        else:
            raise RuntimeError(f"Only Torch tensor or Numpy array is supported. Received bboxes of type {str(type(bboxes))}")


def xyxy_to_yxyx_inplace(bboxes, image_shape: Tuple[int, int]):
    x1x2 = bboxes[..., 0:3:2]
    y1y2 = bboxes[..., 1:4:2]

    sum = x1x2 + y1y2  # Swap via sum
    bboxes[..., 0:3:2] = sum - x1x2
    bboxes[..., 1:4:2] = sum - y1y2
    return bboxes


class YXYXCoordinateFormat(BoundingBoxFormat):
    """
    Bounding boxes format Y1, X1, Y2, X1
    """

    def __init__(self):
        super().__init__()
        self.format = "yxyx"
        self.normalized = False

    def get_to_xyxy(self, inplace: bool):
        if inplace:
            return xyxy_to_yxyx_inplace
        else:
            return xyxy_to_yxyx

    def get_from_xyxy(self, inplace: bool):
        # XYXY <-> YXYX is interchangable operation, so we may reuse same routine here
        if inplace:
            return xyxy_to_yxyx_inplace
        else:
            return xyxy_to_yxyx
