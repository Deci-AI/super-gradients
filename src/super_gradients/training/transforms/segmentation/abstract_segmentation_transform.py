import abc
from abc import abstractmethod
from typing import List

from super_gradients.training.samples import SegmentationSample

__all__ = ["AbstractSegmentationTransform"]


class AbstractSegmentationTransform(abc.ABC):
    """
    Base class for all transforms for object detection sample augmentation.
    """

    @abstractmethod
    def apply_to_sample(self, sample: SegmentationSample) -> SegmentationSample:
        """
        Apply transformation to given segmentation sample.
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
