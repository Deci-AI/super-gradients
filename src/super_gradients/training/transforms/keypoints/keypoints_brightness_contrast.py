import random
import numpy as np

from typing import Tuple, List

from super_gradients.common.registry import register_transform
from .abstract_keypoints_transform import AbstractKeypointTransform
from super_gradients.training.samples import PoseEstimationSample


@register_transform()
class KeypointsBrightnessContrast(AbstractKeypointTransform):
    """
    Apply brightness and contrast change to the input image using following formula:
    image = (image - mean_brightness) * contrast_gain + mean_brightness + brightness_gain
    Transformation preserves input image dtype. Saturation cast is performed at the end of the transformation.
    """

    def __init__(self, prob: float, brightness_range: Tuple[float, float], contrast_range: Tuple[float, float]):
        """

        :param prob:             Probability to apply the transform.
        :param brightness_range: Tuple of two elements, min and max brightness gain. Represents a relative range of
                                 brightness gain with respect to average image brightness. A brightness gain of 1.0
                                 indicates no change in brightness. Therefore, optimal value for this parameter is
                                 somewhere inside (0, 2) range.
        :param contrast_range:   Tuple of two elements, min and max contrast gain. Effective contrast_gain would be
                                 uniformly sampled from this range. Based on definition of contrast gain, it's optimal
                                 value is somewhere inside (0, 2) range.
        """
        if len(brightness_range) != 2:
            raise ValueError("Brightness range must be a tuple of two elements, got: " + str(brightness_range))
        if len(contrast_range) != 2:
            raise ValueError("Contrast range must be a tuple of two elements, got: " + str(contrast_range))
        super().__init__()
        self.prob = prob
        self.brightness_range = tuple(brightness_range)
        self.contrast_range = tuple(contrast_range)

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        if random.random() < self.prob:
            contrast_gain = random.uniform(self.contrast_range[0], self.contrast_range[1])
            brightness_gain = random.uniform(self.brightness_range[0], self.brightness_range[1])

            input_dtype = sample.image.dtype
            image = sample.image.astype(np.float32)
            mean_brightness = np.mean(image, axis=(0, 1))

            image = (image - mean_brightness) * contrast_gain + mean_brightness * brightness_gain

            # get min & max values for the input_dtype
            min_value = np.iinfo(input_dtype).min
            max_value = np.iinfo(input_dtype).max
            sample.image = np.clip(image, a_min=min_value, a_max=max_value).astype(input_dtype)
        return sample

    def get_equivalent_preprocessing(self) -> List:
        raise RuntimeError(f"{self.__class__} does not have equivalent preprocessing because it is non-deterministic.")
