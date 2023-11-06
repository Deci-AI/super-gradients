import random
from typing import List

import numpy as np

from super_gradients.common.object_names import Processings
from super_gradients.common.registry import register_transform
from super_gradients.training.samples import PoseEstimationSample
from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform()
class KeypointsReverseImageChannels(AbstractKeypointTransform):
    """
    Randomly reverse channel order with given probability.
    Given an image with RGB channels, when applied with probability 1, it returns an image with BGR channels.
    With probability 0.5 there is 50/50 chance to return BGR or RGB image.
    It usually helps to improve model's ability to generalize under different color channels order.
    """

    def __init__(self, prob: float):
        """

        :param prob:             Probability to apply the transform.
        """
        super().__init__()
        self.prob = prob

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        if random.random() < self.prob:
            sample.image = np.ascontiguousarray(sample.image[:, :, ::-1])
        return sample

    def get_equivalent_preprocessing(self) -> List:
        if self.prob < 1:
            raise RuntimeError("Cannot set preprocessing pipeline with randomness. Set prob to 1.")
        return [Processings.ReverseImageChannels]
