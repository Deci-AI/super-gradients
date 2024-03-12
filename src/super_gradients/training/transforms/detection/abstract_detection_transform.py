import abc
import warnings
from abc import abstractmethod
from typing import List

from super_gradients.training.samples import DetectionSample

__all__ = ["AbstractDetectionTransform"]


class AbstractDetectionTransform(abc.ABC):
    """
    Base class for all transforms for object detection sample augmentation.
    """

    def __init__(self, additional_samples_count: int = 0):
        """
        :param additional_samples_count: (int) number of samples that must be extra samples from dataset. Default value is 0.
        """
        self._additional_samples_count = additional_samples_count

    @abstractmethod
    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        """
        Apply transformation to given pose estimation sample.
        Important note - function call may return new object, may modify it in-place.
        This is implementation dependent and if you need to keep original sample intact it
        is recommended to make a copy of it BEFORE passing it to transform.

        :param sample: Input sample to transform.
        :return:       Modified sample (It can be the same instance as input or a new object).
        """
        raise NotImplementedError

    @property
    def additional_samples_count(self) -> int:
        warnings.warn(
            "This property is deprecated and will be removed in the future." "Please use `get_number_of_additional_samples` instead.", DeprecationWarning
        )
        return self.get_number_of_additional_samples()

    def get_number_of_additional_samples(self) -> int:
        """
        Returns number of additional samples required. The default implementation assumes that this number is fixed and deterministic.
        Override in case this is not the case, e.g., you randomly choose to apply MixUp, etc
        """
        return self._additional_samples_count

    @property
    def may_require_additional_samples(self) -> bool:
        """
        Indicates whether additional samples are required. The default implementation assumes that this indicator is fixed and deterministic.
        Override in case this is not the case, e.g., you randomly choose to apply MixUp, etc
        """
        return self._additional_samples_count > 0

    @abstractmethod
    def get_equivalent_preprocessing(self) -> List:
        raise NotImplementedError
