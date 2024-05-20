from typing import List

import numpy as np
from super_gradients.common.registry import register_transform

from .abstract_obb_transform import AbstractOBBDetectionTransform
from super_gradients.training.samples.obb_sample import OBBSample


@register_transform()
class OBBRemoveSmallObjects(AbstractOBBDetectionTransform):
    """
    Remove pose instances from data sample that are too small or have too few visible keypoints.
    """

    def __init__(self, min_size: int, min_area: int):
        """
        :param min_size: Minimum size (width or height) of oriented box to keep in the sample
        :param min_area: Minimum area of oriented box to keep in the sample
        """
        super().__init__()
        self.min_size = min_size
        self.min_area = min_area

    def apply_to_sample(self, sample: OBBSample) -> OBBSample:
        """
        Apply transformation to given pose estimation sample.

        :param sample: Input sample to transform.
        :return:       Filtered sample.
        """
        mask = np.ones(len(sample), dtype=bool)
        if self.min_size:
            min_size_mask = sample.rboxes_cxcywhr[:, 2:4].min(axis=1) >= self.min_size
            mask &= min_size_mask
        if self.min_area:
            min_area_mask = sample.rboxes_cxcywhr[:, 2] * sample.rboxes_cxcywhr[:, 3] >= self.min_area
            mask &= min_area_mask
        return sample.filter_by_mask(mask)

    def __repr__(self):
        return self.__class__.__name__ + (f"(min_size={self.min_size}, " f"min_area={self.min_area})")

    def get_equivalent_preprocessing(self) -> List:
        return []
