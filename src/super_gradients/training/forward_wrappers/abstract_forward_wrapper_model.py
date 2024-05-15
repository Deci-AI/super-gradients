import abc
import torch
from torch import nn


class AbstractForwardWrapperModel(abc.ABC):
    def __init__(self, model: nn.Module = None):
        """
        Initialize the AbstractForwardWrapperModel with an optional PyTorch model.

        :param model: An instance of nn.Module to be wrapped by this class, default is None.
        """
        self.model = model

    @abc.abstractmethod
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to be implemented by subclasses that defines the forward pass.

        :param inputs: A torch.Tensor containing the input to the model.
        :return: A torch.Tensor containing the model's output.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def set_model(self, model: nn.Module):
        """
        Set the model for this wrapper.

        :param model: An instance of nn.Module to be used by this wrapper.
        """
        self.model = model
