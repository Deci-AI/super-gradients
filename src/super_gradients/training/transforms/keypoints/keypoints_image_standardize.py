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

    :attr max_value: Current maximum value of the image pixels. (usually 255)
    """

    def __init__(self, max_value: float = 255.0):
        super().__init__()
        self.max_value = max_value

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        sample.image = (sample.image / self.max_value).astype(np.float32)
        return sample

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.StandardizeImage: {"max_value": self.max_value}}]

    def __repr__(self):
        return self.__class__.__name__ + f"(max_value={self.max_value})"
