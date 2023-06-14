import copy
import math
from typing import List

import torch
from torch import nn


def _replace_activations_recursive(module: nn.Module, new_activation: nn.Module, activations_to_replace: List[type]):
    """
    A helper called in replace_activations(...)
    """
    for n, m in module.named_children():
        if type(m) in activations_to_replace:
            setattr(module, n, copy.deepcopy(new_activation))
        else:
            _replace_activations_recursive(m, new_activation, activations_to_replace)


def replace_activations(module: nn.Module, new_activation: nn.Module, activations_to_replace: List[type]):
    """
    Recursively go through module and replaces each activation in activations_to_replace with a copy of new_activation
    :param module:                  a module that will be changed inplace
    :param new_activation:          a sample of a new activation (will be copied)
    :param activations_to_replace:  types of activations to replace, each must be a subclass of nn.Module
    """
    # check arguments once before the recursion
    assert isinstance(new_activation, nn.Module), "new_activation should be nn.Module"
    assert all(
        [isinstance(t, type) and issubclass(t, nn.Module) for t in activations_to_replace]
    ), "activations_to_replace should be types that are subclasses of nn.Module"

    # do the replacement
    _replace_activations_recursive(module, new_activation, activations_to_replace)


class NormalizationAdapter(torch.nn.Module):
    """
    Denormalizes input by mean_original, std_original, then normalizes by mean_required, std_required.

    Used in KD training where teacher expects data normalized by mean_required, std_required.

    mean_original, std_original, mean_required, std_required are all list-like objects of length that's equal to the
     number of input channels.

    """

    def __init__(self, mean_original, std_original, mean_required, std_required):
        super(NormalizationAdapter, self).__init__()
        mean_original = torch.tensor(mean_original).unsqueeze(-1).unsqueeze(-1)
        std_original = torch.tensor(std_original).unsqueeze(-1).unsqueeze(-1)
        mean_required = torch.tensor(mean_required).unsqueeze(-1).unsqueeze(-1)
        std_required = torch.tensor(std_required).unsqueeze(-1).unsqueeze(-1)

        self.additive = torch.nn.Parameter((mean_original - mean_required) / std_original)
        self.multiplier = torch.nn.Parameter(std_original / std_required)

    def forward(self, x):
        x = (x + self.additive) * self.multiplier
        return x


def width_multiplier(original, factor, divisor: int = None):
    if divisor is None:
        return int(original * factor)
    else:
        return math.ceil(int(original * factor) / divisor) * divisor


def autopad(kernel, padding=None):
    # PAD TO 'SAME'
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
    return padding
