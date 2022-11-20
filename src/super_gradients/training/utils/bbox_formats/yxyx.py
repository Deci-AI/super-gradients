import torch

from super_gradients.training.utils.bbox_formats.bbox_format import BoundingBoxFormat
from typing import Union, Tuple

import numpy as np
from torch import Tensor


def xyxy2yxyx(bboxes: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    if torch.is_tensor(bboxes):
        return bboxes[..., torch.tensor([1, 0, 3, 2], dtype=torch.long, device=bboxes.device)]
    elif isinstance(bboxes, np.ndarray):
        return bboxes[..., np.array([1, 0, 3, 2], dtype=int)]
    else:
        raise RuntimeError(f"Only Torch tensor or Numpy array is supported. Received bboxes of type {str(type(bboxes))}")


def xyxy2yxyx_inplace(bboxes: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
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

    def to_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        if inplace:
            return xyxy2yxyx_inplace(bboxes)
        else:
            return xyxy2yxyx(bboxes)

    def from_xyxy(self, bboxes: Union[Tensor, np.ndarray], image_shape: Tuple[int, int], inplace: bool) -> Union[Tensor, np.ndarray]:
        # XYXY <-> YXYX is interchangable operation, so we may reuse same routine here
        return self.to_xyxy(bboxes, image_shape, inplace)
