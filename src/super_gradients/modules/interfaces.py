import abc
from typing import Callable

from torch import nn

__all__ = ["SupportsReplaceNumClasses"]


class SupportsReplaceNumClasses:
    """
    Protocol interface for modules that support replacing the number of classes.
    Derived classes should implement the `replace_num_classes` method.

    This interface class serves a purpose of explicitly indicating whether a class supports optimized head replacement:

    >>> class PredictionHead(nn.Module, SupportsReplaceNumClasses):
    >>>    def replace_num_classes(self, num_classes: int, compute_new_weights_fn: Callable[[nn.Module, int], nn.Module] = None):
    >>>       ...
    """

    @abc.abstractmethod
    def replace_num_classes(self, num_classes: int, compute_new_weights_fn: Callable[[nn.Module, int], nn.Module]):
        """
        Replace the number of classes in the module.

        :param num_classes: New number of classes.
        :param compute_new_weights_fn: (callable) An optional function that computes the new weights for the new classes.
            It takes existing nn.Module and returns a new one.
        :return: None
        """
        raise NotImplementedError
