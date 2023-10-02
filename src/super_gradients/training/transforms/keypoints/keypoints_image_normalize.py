from typing import List, Union

import numpy as np
from omegaconf import ListConfig

from super_gradients.common.object_names import Transforms, Processings
from super_gradients.common.registry import register_transform
from super_gradients.training.samples import PoseEstimationSample

from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform(Transforms.KeypointsImageNormalize)
class KeypointsImageNormalize(AbstractKeypointTransform):
    """
    Normalize image with mean and std.
    """

    def __init__(self, mean: Union[float, List[float], ListConfig], std: Union[float, List[float], ListConfig]):
        super().__init__()

        self.mean = np.array(list(mean)).reshape((1, 1, -1)).astype(np.float32)
        self.std = np.array(list(std)).reshape((1, 1, -1)).astype(np.float32)

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        sample.image = (sample.image - self.mean) / self.std
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"

    def get_equivalent_preprocessing(self) -> List:
        return [{Processings.NormalizeImage: {"mean": self.mean, "std": self.std}}]
