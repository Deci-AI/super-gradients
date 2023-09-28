import abc
from abc import abstractmethod
from typing import List

from super_gradients.training.samples import PoseEstimationSample


class AbstractKeypointTransform(abc.ABC):
    """
    Base class for all transforms for keypoints augmentation.
    All transforms subclassing it should implement __call__ method which takes image, mask and keypoints as input and
    returns transformed image, mask and keypoints.

    :attr additional_samples_count: Number of additional samples to generate for each image.
                                    This property is used for mixup & mosaic transforms that needs an extra samples.
    """

    def __init__(self, additional_samples_count: int = 0):
        """
        :param additional_samples_count: (int) number of samples that must be extra samples from dataset. Default value is 0.
        """
        self.additional_samples_count = additional_samples_count

    @abstractmethod
    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
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
