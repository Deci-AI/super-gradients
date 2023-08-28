import random
from typing import Optional, Tuple

import numpy as np

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry import register_transform
from super_gradients.training.transforms.keypoint_transforms import KeypointTransform
from super_gradients.training.transforms.transforms import augment_hsv

logger = get_logger(__name__)


@register_transform()
class KeypointsHSV(KeypointTransform):
    """
    Apply color change in HSV color space to the input image.

    :param prob:            Probability to apply the transform.
    :param hgain:           Hue gain.
    :param sgain:           Saturation gain.
    :param vgain:           Value gain.
    """

    def __init__(self, prob: float, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5):
        super().__init__()
        self.prob = prob
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(
        self, image: np.ndarray, mask: np.ndarray, joints: np.ndarray, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        if image.shape[2] != 3:
            raise ValueError("HSV transform expects image with 3 channels, got: " + str(image.shape[2]))

        if random.random() < self.prob:
            image_copy = image.copy()
            augment_hsv(image_copy, self.hgain, self.sgain, self.vgain, bgr_channels=(0, 1, 2) if random.random() < 0.5 else (2, 1, 0))
            image = image_copy
        return image, mask, joints, areas, bboxes
