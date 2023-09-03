from typing import Callable, Optional, TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    # This is a hack to avoid circular imports while still having type hints.
    from super_gradients.training.processing.processing import Processing


class HasPreprocessingParams:
    """
    Protocol interface for torch datasets that support getting preprocessing params, later to be passed to a model
    that obeys NeedsPreprocessingParams. This interface class serves a purpose of explicitly indicating whether a torch dataset has
    get_dataset_preprocessing_params implemented.

    """

    def get_dataset_preprocessing_params(self):
        raise NotImplementedError(f"get_dataset_preprocessing_params is not implemented in the derived class {self.__class__.__name__}")


class HasPredict:
    """
    Protocol class serves a purpose of explicitly indicating whether a torch model has the functionality of ".predict"
    as defined in SG.

    """

    def set_dataset_processing_params(self, *args, **kwargs):
        """Set the processing parameters for the dataset."""
        raise NotImplementedError(f"set_dataset_processing_params is not implemented in the derived class {self.__class__.__name__}")

    def predict(self, images, *args, **kwargs):
        raise NotImplementedError(f"predict is not implemented in the derived class {self.__class__.__name__}")

    def predict_webcam(self, *args, **kwargs):
        raise NotImplementedError(f"predict_webcam is not implemented in the derived class {self.__class__.__name__}")

    def get_input_channels(self) -> int:
        """
        Get the number of input channels for the model.
        :return: (int) Number of input channels.
        """
        raise NotImplementedError(f"get_input_channels is not implemented in the derived class {self.__class__.__name__}")

    def get_processing_params(self) -> Optional["Processing"]:
        raise NotImplementedError(f"get_processing_params is not implemented in the derived class {self.__class__.__name__}")


class SupportsReplaceNumClasses:
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
        raise NotImplementedError(f"replace_num_classes is not implemented in the derived class {self.__class__.__name__}")
