from abc import ABC
from typing import Callable, Optional, TYPE_CHECKING, Dict

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


class SupportsReplaceInputChannels(ABC):
    """
    Protocol interface for modules that support replacing the number of input channels.
    Derived classes should implement the `replace_input_channels` method.

    This interface class serves the purpose of explicitly indicating whether a class supports optimized input channel replacement:

    >>> class InputLayer(nn_Module, SupportsReplaceInputChannels):
    >>>    def replace_input_channels(self, in_channels: int, compute_new_weights_fn: Callable[[nn.Module, int], nn.Module] = None):
    >>>       ...

    """

    def replace_input_channels(self, in_channels: int, compute_new_weights_fn: Optional[Callable[[nn.Module, int], nn.Module]]):
        """
        Replace the number of input channels in the module.

        :param in_channels:             New number of input channels.
        :param compute_new_weights_fn:  (Optional) function that computes the new weights for the new input channels.
                                        It takes the existing nn_Module and returns a new one.
        """
        raise NotImplementedError(f"`replace_input_channels` is not implemented in the derived class `{self.__class__.__name__}`")

    def get_input_channels(self) -> int:
        """Get the number of input channels for the model.

        :return: Number of input channels.
        """
        raise NotImplementedError(f"`get_input_channels` is not implemented in the derived class `{self.__class__.__name__}`")


class SupportsFineTune:
    def get_finetune_lr_dict(self, lr: float) -> Dict[str, float]:
        """
        Returns a dictionary, mapping lr to the unfrozen part of the network, in the same fashion as using initial_lr in trianing_params
         when calling Trainer.train().
        For example:
            def get_finetune_lr_dict(self, lr: float) -> Dict[str, float]:
                return {"default": 0, "head": lr}

        :param lr: float, learning rate for the part of the network to be tuned.
        :return: learning rate mapping that can be used by
         super_gradients.training.utils.optimizer_utils.initialize_param_groups
        """
        raise NotImplementedError("Fine tune is not supported for this model")
