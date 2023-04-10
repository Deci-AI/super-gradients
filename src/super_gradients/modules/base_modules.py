from abc import abstractmethod, ABC
from typing import Union, List

from torch import nn

__all__ = ["BaseDetectionModule"]


class BaseDetectionModule(nn.Module, ABC):
    """
    An interface for a module that is easy to integrate into a model with complex connections
    """

    def __init__(self, in_channels: Union[List[int], int], **kwargs):
        """
        :param in_channels: defines channels of tensor(s) that will be accepted by a module in forward
        """
        super().__init__()
        self.in_channels = in_channels

    @property
    @abstractmethod
    def out_channels(self) -> Union[List[int], int]:
        """
        :return: channels of tensor(s) that will be returned by a module  in forward
        """
        raise NotImplementedError()
