from abc import abstractmethod, ABC
from typing import Union, List


from torch import nn


class BaseDetectionModule(nn.Module, ABC):

    def __init__(self, in_channels: Union[List[int], int]):
        """
        :param in_channels:
        """
        super().__init__()
        self.in_channels = in_channels

    @property
    @abstractmethod
    def out_channels(self) -> Union[List[int], int]:
        raise NotImplementedError()


ALL_DETECTION_MODULES = {}
