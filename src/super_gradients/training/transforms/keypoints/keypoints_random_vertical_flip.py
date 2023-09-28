import random

import numpy as np

from super_gradients.common.object_names import Transforms
from super_gradients.common.registry.registry import register_transform
from super_gradients.training.samples import PoseEstimationSample
from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform(Transforms.KeypointsRandomVerticalFlip)
class KeypointsRandomVerticalFlip(AbstractKeypointTransform):
    """
    Flip image, mask and joints vertically with a given probability.
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        if sample.image.shape[:2] != sample.mask.shape[:2]:
            raise RuntimeError(f"Image shape ({sample.image.shape[:2]}) does not match mask shape ({sample.mask.shape[:2]}).")

        if random.random() < self.prob:
            sample.image = self.apply_to_image(sample.image)
            sample.mask = self.apply_to_image(sample.mask)

            rows, cols = sample.image.shape[:2]
            sample.joints = self.apply_to_keypoints(sample.joints, rows)

            if sample.bboxes is not None:
                sample.bboxes = self.apply_to_bboxes(sample.bboxes, rows)

        return sample

    def apply_to_image(self, image):
        return np.ascontiguousarray(np.flipud(image))

    def apply_to_keypoints(self, keypoints, rows):
        keypoints = keypoints.copy()
        keypoints[:, :, 1] = rows - keypoints[:, :, 1] - 1
        return keypoints

    def apply_to_bboxes(self, bboxes, rows):
        bboxes = bboxes.copy()
        bboxes[:, 1] = rows - (bboxes[:, 1] + bboxes[:, 3]) - 1
        return bboxes

    def __repr__(self):
        return self.__class__.__name__ + f"(prob={self.prob})"

    def get_equivalent_preprocessing(self):
        raise RuntimeError(f"{self.__class__} does not have equivalent preprocessing because it is non-deterministic.")
