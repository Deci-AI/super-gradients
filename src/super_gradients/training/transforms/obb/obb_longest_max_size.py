import random
from typing import List

import cv2
import numpy as np
from super_gradients.common.object_names import Processings
from super_gradients.common.registry import register_transform
from super_gradients.training.transforms.utils import _rescale_bboxes

from super_gradients.training.samples.obb_sample import OBBSample
from .abstract_obb_transform import AbstractOBBDetectionTransform


@register_transform()
class OBBDetectionLongestMaxSize(AbstractOBBDetectionTransform):
    """
    Resize data sample to guarantee that input image dimensions is not exceeding maximum width & height
    """

    def __init__(self, max_height: int, max_width: int, interpolation: int = cv2.INTER_LINEAR, prob: float = 1.0):
        """

        :param max_height: (int) Maximum image height
        :param max_width: (int)  Maximum image width
        :param interpolation:    Used interpolation method for image
        :param prob:             Probability of applying this transform. Default: 1.0
        """
        super().__init__()
        self.max_height = int(max_height)
        self.max_width = int(max_width)
        self.interpolation = int(interpolation)
        self.prob = float(prob)

    def apply_to_sample(self, sample: OBBSample) -> OBBSample:
        if random.random() < self.prob:
            height, width = sample.image.shape[:2]
            scale = min(self.max_height / height, self.max_width / width)

            sample = OBBSample(
                image=self.apply_to_image(sample.image, scale, cv2.INTER_LINEAR),
                rboxes_cxcywhr=self.apply_to_bboxes(sample.rboxes_cxcywhr, scale),
                labels=sample.labels,
                is_crowd=sample.is_crowd,
                additional_samples=None,
            )

            if sample.image.shape[0] != self.max_height and sample.image.shape[1] != self.max_width:
                raise RuntimeError(f"Image shape is not as expected (scale={scale}, input_shape={height, width}, resized_shape={sample.image.shape[:2]})")

            if sample.image.shape[0] > self.max_height or sample.image.shape[1] > self.max_width:
                raise RuntimeError(f"Image shape is not as expected (scale={scale}, input_shape={height, width}, resized_shape={sample.image.shape[:2]}")

        return sample

    @classmethod
    def apply_to_image(cls, image: np.ndarray, scale: float, interpolation: int) -> np.ndarray:
        height, width = image.shape[:2]

        if scale != 1.0:
            new_height, new_width = tuple(int(dim * scale + 0.5) for dim in (height, width))
            image = cv2.resize(image, dsize=(new_width, new_height), interpolation=interpolation)
        return image

    @classmethod
    def apply_to_bboxes(cls, bboxes: np.ndarray, scale: float) -> np.ndarray:
        return _rescale_bboxes(bboxes, (scale, scale))

    def get_equivalent_preprocessing(self) -> List:
        return [{Processings.OBBDetectionLongestMaxSizeRescale: {"output_shape": (self.max_height, self.max_width)}}]
