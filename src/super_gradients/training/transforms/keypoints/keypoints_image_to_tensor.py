from typing import List

import numpy as np
import torch

from super_gradients.common.object_names import Transforms, Processings
from super_gradients.common.registry import register_transform
from super_gradients.training.samples import PoseEstimationSample

from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform(Transforms.KeypointsImageToTensor)
class KeypointsImageToTensor(AbstractKeypointTransform):
    """
    Convert image from numpy array to tensor and permute axes to [C,H,W].
    """

    def __init__(self):
        super().__init__()

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        sample.image = torch.from_numpy(np.transpose(sample.image, (2, 0, 1))).float()
        return sample

    def get_equivalent_preprocessing(self) -> List:
        return [
            {Processings.ImagePermute: {"permutation": (2, 0, 1)}},
        ]

    def __repr__(self):
        return self.__class__.__name__ + "()"
