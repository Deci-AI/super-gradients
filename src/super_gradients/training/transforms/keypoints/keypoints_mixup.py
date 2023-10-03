import random
import numpy as np

from typing import Optional

from super_gradients.common.registry import register_transform
from super_gradients.training.samples import PoseEstimationSample

from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform()
class KeypointsMixup(AbstractKeypointTransform):
    """
    Mix two samples together.

    :param prob:            Probability to apply the transform.
    """

    def __init__(self, prob: float):
        """

        :param prob:            Probability to apply the transform.
        """
        super().__init__(additional_samples_count=1)
        self.prob = prob

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        if random.random() < self.prob:
            sample = self.apply_mixup(sample, sample.additional_samples[0])
        return sample

    def apply_mixup(self, sample: PoseEstimationSample, other: PoseEstimationSample) -> PoseEstimationSample:
        sample.image = (sample.image * 0.5 + other.image * 0.5).astype(sample.image)
        sample.mask = np.logical_or(sample.mask, other.mask).astype(sample.mask.dtype)
        sample.joints = np.concatenate([sample.joints, other.joints], axis=0)
        sample.is_crowd = np.concatenate([sample.is_crowd, other.is_crowd], axis=0)

        sample.bboxes = self._concatenate_arrays(sample.bboxes, other.bboxes, (0, 4))
        sample.areas = self._concatenate_arrays(sample.areas, other.areas, (0,))
        sample.additional_samples = None
        return sample

    def _concatenate_arrays(self, arr1: Optional[np.ndarray], arr2: Optional[np.ndarray], shape_if_empty):
        if arr1 is None:
            arr1 = np.zeros(shape_if_empty, dtype=np.float32)
        if arr2 is None:
            arr2 = np.zeros(shape_if_empty, dtype=np.float32)
        return np.concatenate([arr1, arr2], axis=0)

    def get_equivalent_preprocessing(self):
        raise RuntimeError(f"{self.__class__} does not have equivalent preprocessing because it is non-deterministic.")
