import abc
from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from super_gradients.training.samples import PoseEstimationSample


class AbstractKeypointTransform(abc.ABC):
    """
    Base class for all transforms for keypoints augmentation.
    All transforms subclassing it should implement __call__ method which takes image, mask and keypoints as input and
    returns transformed image, mask and keypoints.

    :param additional_samples_count: Number of additional samples to generate for each image.
                                    This property is used for mixup & mosaic transforms that needs an extra samples.
    """

    def __init__(self, additional_samples_count: int = 0):
        """
        :param additional_samples_count: (int) number of samples that must be extra samples from dataset. Default value is 0.
        """
        self.additional_samples_count = additional_samples_count

    def __call__(
        self, image: np.ndarray, mask: np.ndarray, joints: np.ndarray, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply transformation to pose estimation sample passed as a tuple
        This method acts as a wrapper for apply_to_sample method to support old-style API.
        """
        sample = PoseEstimationSample(
            image=image,
            mask=mask,
            joints=joints,
            areas=areas,
            bboxes_xywh=bboxes,
            is_crowd=np.zeros(len(joints)),  # Old style API does not pass is_crowd parameter, so we set it to zeros
            additional_samples=None,
        )
        sample = self.apply_to_sample(sample)
        return sample.image, sample.mask, sample.joints, sample.areas, sample.bboxes_xywh

    @abstractmethod
    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        """
        Apply transformation to given pose estimation sample.
        Important note - function call may return new object, may modify it in-place.
        This is implementation dependent and if you need to keep original sample intact it
        is recommended to make a copy of it BEFORE passing it to transform.

        :param sample: Input sample to transform.
        :return:       Modified sample (It can be the same instance as input or a new object).
        """
        raise NotImplementedError

    @abstractmethod
    def get_equivalent_preprocessing(self) -> List:
        raise NotImplementedError


KeypointTransform = AbstractKeypointTransform  # Type alias for backward compatibility
