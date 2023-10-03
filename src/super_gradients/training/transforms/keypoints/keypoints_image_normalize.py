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
    Normalize image with mean and std using formula `(image - mean) / std`.
    Output image will allways have dtype of np.float32.
    """

    def __init__(self, mean: Union[float, List[float], ListConfig], std: Union[float, List[float], ListConfig]):
        """

        :param mean: (float, List[float]) A constant bias to be subtracted from the image.
                     If it is a list, it should have the same length as the number of channels in the image.
        :param std:  (float, List[float]) A scaling factor to be applied to the image after subtracting mean.
                     If it is a list, it should have the same length as the number of channels in the image.
        """
        super().__init__()
        self.mean = np.array(list(mean)).reshape((1, 1, -1)).astype(np.float32)
        self.std = np.array(list(std)).reshape((1, 1, -1)).astype(np.float32)

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        """
        Apply transformation to given pose estimation sample

        :param sample: A pose estimation sample
        :return:       Same pose estimation sample with normalized image
        """
        sample.image = np.divide(sample.image - self.mean, self.std, dtype=np.float32)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"

    def get_equivalent_preprocessing(self) -> List:
        return [{Processings.NormalizeImage: {"mean": self.mean, "std": self.std}}]
