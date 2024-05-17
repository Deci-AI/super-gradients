from typing import List, Dict

import numpy as np
from super_gradients.common.object_names import Processings
from super_gradients.common.registry import register_transform
from .obb_sample import OBBSample
from .abstract_obb_transform import AbstractOBBDetectionTransform


@register_transform()
class OBBDetectionStandardize(AbstractOBBDetectionTransform):
    """
    Standardize image pixel values with img/max_val

    :param max_val: Current maximum value of the image pixels. (usually 255)
    """

    def __init__(self, max_value: float = 255.0):
        super().__init__()
        self.max_value = float(max_value)

    @classmethod
    def apply_to_image(self, image: np.ndarray, max_value: float) -> np.ndarray:
        return (image / max_value).astype(np.float32)

    def apply_to_sample(self, sample: OBBSample) -> OBBSample:
        sample.image = self.apply_to_image(sample.image, max_value=self.max_value)
        return sample

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.StandardizeImage: {"max_value": self.max_value}}]
