from typing import List, Dict

import numpy as np

from super_gradients.common.object_names import Transforms, Processings
from super_gradients.common.registry import register_transform
from super_gradients.training.samples import PoseEstimationSample
from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform(Transforms.KeypointsImageStandardize)
class KeypointsImageStandardize(AbstractKeypointTransform):
    """
    Standardize image pixel values with img/max_value formula.
    Output image will allways have dtype of np.float32.

    :param max_value: Current maximum value of the image pixels. (usually 255)
    """

    def __init__(self, max_value: float = 255.0):
        """

        :param max_value: A constant value to divide the image by.
        """
        super().__init__()
        self.max_value = max_value

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        """
        Apply transformation to given pose estimation sample

        :param sample: A pose estimation sample
        :return:       Same pose estimation sample with standardized image
        """
        sample.image = np.divide(sample.image, self.max_value, dtype=np.float32)
        return sample

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.StandardizeImage: {"max_value": self.max_value}}]

    def __repr__(self):
        return self.__class__.__name__ + f"(max_value={self.max_value})"
