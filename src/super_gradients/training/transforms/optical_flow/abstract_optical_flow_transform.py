import abc

from super_gradients.training.samples import OpticalFlowSample


class AbstractOpticalFlowTransform(abc.ABC):
    """
    Base class for all transforms for optical flow sample augmentation.
    """

    @abc.abstractmethod
    def __call__(self, sample: OpticalFlowSample) -> OpticalFlowSample:
        """
        Apply transformation to given optical flow sample.
        Important note - function call may return new object, may modify it in-place.
        This is implementation dependent and if you need to keep original sample intact it
        is recommended to make a copy of it BEFORE passing it to transform.

        :param sample: Input sample to transform.
        :return:       Modified sample (It can be the same instance as input or a new object).
        """
        raise NotImplementedError()
