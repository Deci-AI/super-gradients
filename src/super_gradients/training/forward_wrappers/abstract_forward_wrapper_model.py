import abc
import torch


class AbstractForwardWrapperModel(abc.ABC):
    @abc.abstractmethod
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to be implemented by subclasses that defines the forward pass.

        :param inputs: A torch.Tensor containing the input to the model.
        :return: A torch.Tensor containing the model's output.
        """
        raise NotImplementedError("Subclasses must implement this method")
