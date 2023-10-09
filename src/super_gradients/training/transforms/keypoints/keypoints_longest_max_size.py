import random
from typing import List

import cv2
import numpy as np

from super_gradients.common.object_names import Transforms, Processings
from super_gradients.common.registry import register_transform
from super_gradients.training.samples import PoseEstimationSample
from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform(Transforms.KeypointsLongestMaxSize)
class KeypointsLongestMaxSize(AbstractKeypointTransform):
    """
    Resize data sample to guarantee that input image dimensions is not exceeding maximum width & height
    """

    def __init__(self, max_height: int, max_width: int, interpolation: int = cv2.INTER_LINEAR, prob: float = 1.0):
        """

        :param max_height: (int) - Maximum image height
        :param max_width: (int) - Maximum image width
        :param interpolation: Used interpolation method for image
        :param prob: Probability of applying this transform
        """
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.interpolation = interpolation
        self.prob = prob

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        if random.random() < self.prob:
            height, width = sample.image.shape[:2]
            scale = min(self.max_height / height, self.max_width / width)
            sample.image = self.apply_to_image(sample.image, scale, cv2.INTER_LINEAR)
            sample.mask = self.apply_to_image(sample.mask, scale, cv2.INTER_NEAREST)

            if sample.image.shape[0] != self.max_height and sample.image.shape[1] != self.max_width:
                raise RuntimeError(f"Image shape is not as expected (scale={scale}, input_shape={height, width}, resized_shape={sample.image.shape[:2]})")

            if sample.image.shape[0] > self.max_height or sample.image.shape[1] > self.max_width:
                raise RuntimeError(f"Image shape is not as expected (scale={scale}, input_shape={height, width}, resized_shape={sample.image.shape[:2]}")

            sample.joints = self.apply_to_keypoints(sample.joints, scale)
            if sample.bboxes_xywh is not None:
                sample.bboxes_xywh = self.apply_to_bboxes(sample.bboxes_xywh, scale)

            if sample.areas is not None:
                sample.areas = np.multiply(sample.areas, scale**2, dtype=np.float32)

        return sample

    @classmethod
    def apply_to_image(cls, img, scale, interpolation):
        height, width = img.shape[:2]

        if scale != 1.0:
            new_height, new_width = tuple(int(dim * scale + 0.5) for dim in (height, width))
            img = cv2.resize(img, dsize=(new_width, new_height), interpolation=interpolation)
        return img

    @classmethod
    def apply_to_keypoints(cls, keypoints, scale):
        keypoints = keypoints.astype(np.float32, copy=True)
        keypoints[:, :, 0:2] *= scale
        return keypoints

    @classmethod
    def apply_to_bboxes(cls, bboxes, scale):
        return np.multiply(bboxes, scale, dtype=np.float32)

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(max_height={self.max_height}, "
            f"max_width={self.max_width}, "
            f"interpolation={self.interpolation}, prob={self.prob})"
        )

    def get_equivalent_preprocessing(self) -> List:
        return [{Processings.KeypointsLongestMaxSizeRescale: {"output_shape": (self.max_height, self.max_width)}}]
