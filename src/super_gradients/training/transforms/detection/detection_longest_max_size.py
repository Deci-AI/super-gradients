import random
from typing import List

import cv2
import numpy as np

from super_gradients.common.object_names import Transforms, Processings
from super_gradients.common.registry import register_transform
from super_gradients.training.samples import DetectionSample
from .abstract_detection_transform import AbstractDetectionTransform
from .legacy_detection_transform_mixin import LegacyDetectionTransformMixin


@register_transform(Transforms.DetectionLongestMaxSize)
class DetectionLongestMaxSize(AbstractDetectionTransform, LegacyDetectionTransformMixin):
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

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        if random.random() < self.prob:
            height, width = sample.image.shape[:2]
            scale = min(self.max_height / height, self.max_width / width)

            sample = DetectionSample(
                image=self.apply_to_image(sample.image, scale, cv2.INTER_LINEAR),
                bboxes_xyxy=self.apply_to_bboxes(sample.bboxes_xyxy, scale),
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
        return np.multiply(bboxes, scale, dtype=np.float32)

    def get_equivalent_preprocessing(self) -> List:
        return [{Processings.DetectionLongestMaxSizeRescale: {"output_shape": (self.max_height, self.max_width)}}]
