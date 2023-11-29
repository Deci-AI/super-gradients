import dataclasses
from typing import Union

import numpy as np
import torch
from PIL import Image

__all__ = ["SegmentationSample"]


@dataclasses.dataclass
class SegmentationSample:
    """
    A data class describing a single object detection sample that comes from a dataset.
    It contains both input image and target information to train an object detection model.

    :param image:              Associated image with a sample. Can be in [H,W,C] or [C,H,W] format
    :param bboxes_xywh:        Numpy array of [N,4] shape with bounding box of each instance (XYWH)
    :param labels:             Numpy array of [N] shape with class label for each instance
    :param is_crowd:           (Optional) Numpy array of [N] shape with is_crowd flag for each instance
    :param additional_samples: (Optional) List of additional samples for the same image.
    """

    __slots__ = ["image", "mask"]

    image: Image
    mask: Image

    def __init__(self, image: Union[np.ndarray, torch.Tensor], mask: Union[np.ndarray, torch.Tensor]):
        self.image = image
        self.mask = mask
