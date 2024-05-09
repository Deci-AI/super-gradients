import abc

import torch
from torch import nn
from abc import abstractmethod


class AbstractForwardWrapperModel(abc.ABC):
    @abstractmethod
    def __call__(self, images: torch.Tensor, model: nn.Module, **kwargs):
        raise NotImplementedError
