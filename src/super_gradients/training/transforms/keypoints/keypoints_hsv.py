import random

from super_gradients.common.registry import register_transform
from super_gradients.training.transforms.transforms import augment_hsv
from super_gradients.training.samples import PoseEstimationSample

from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform()
class KeypointsHSV(AbstractKeypointTransform):
    """
    Apply color change in HSV color space to the input image.

    :param prob:            Probability to apply the transform.
    :param hgain:           Hue gain.
    :param sgain:           Saturation gain.
    :param vgain:           Value gain.
    """

    def __init__(self, prob: float, hgain: float, sgain: float, vgain: float):
        """

        :param prob:            Probability to apply the transform.
        :param hgain:           Hue gain.
        :param sgain:           Saturation gain.
        :param vgain:           Value gain.
        """
        super().__init__()
        self.prob = prob
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        if sample.image.shape[2] != 3:
            raise ValueError("HSV transform expects image with 3 channels, got: " + str(sample.image.shape[2]))

        if random.random() < self.prob:
            image_copy = sample.image.copy()
            augment_hsv(image_copy, self.hgain, self.sgain, self.vgain, bgr_channels=(0, 1, 2))
            sample.image = image_copy
        return sample

    def get_equivalent_preprocessing(self):
        raise RuntimeError(f"{self.__class__} does not have equivalent preprocessing because it is non-deterministic.")
