from typing import Callable

from torch import nn
from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class HasPreprocessingParams(Protocol):
    """
    Protocol interface for torch datasets that support getting preprocessing params, later to be passed to a model
    that obeys NeedsPreprocessingParams. This interface class serves a purpose of explicitly indicating whether a torch dataset has
    get_dataset_preprocessing_params implemented.

    """

    def get_dataset_preprocessing_params(self):
        ...


@runtime_checkable
class HasPredict(Protocol):
    """
    Protocol class serves a purpose of explicitly indicating whether a torch model has the functionality of ".predict"
    as defined in SG.

    """

    def set_dataset_processing_params(self, *args, **kwargs):
        """Set the processing parameters for the dataset."""
        ...

    def predict(self, images, *args, **kwargs):
        ...

    def predict_webcam(self, *args, **kwargs):
        ...


@runtime_checkable
class SupportsReplaceNumClasses(Protocol):
    """
    Protocol interface for modules that support replacing the number of classes.
    Derived classes should implement the `replace_num_classes` method.

    This interface class serves a purpose of explicitly indicating whether a class supports optimized head replacement:

    >>> class PredictionHead(nn.Module, SupportsReplaceNumClasses):
    >>>    def replace_num_classes(self, num_classes: int, compute_new_weights_fn: Callable[[nn.Module, int], nn.Module] = None):
    >>>       ...
    """

    def replace_num_classes(self, num_classes: int, compute_new_weights_fn: Callable[[nn.Module, int], nn.Module]):
        """
        Replace the number of classes in the module.

        :param num_classes: New number of classes.
        :param compute_new_weights_fn: (callable) An optional function that computes the new weights for the new classes.
            It takes existing nn.Module and returns a new one.
        :return: None
        """
        ...
