import abc

import torch
from torch import nn
from abc import abstractmethod


class AbstractForwardWrapperModel(abc.ABC):
    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def __call__(self, inputs: torch.Tensor):
        raise NotImplementedError
