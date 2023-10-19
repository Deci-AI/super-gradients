from typing import Union

import torch
from torch import nn

__all__ = ["replace_in_channels_with_random_weights"]


def replace_in_channels_with_random_weights(module: Union[nn.Conv2d, nn.Linear, nn.Module], in_channels: int) -> nn.Module:
    """
    Replace the input channels in the module with random weights.
    This is useful for replacing the input layer of a model.
    This implementation supports Conv2d layers.
    Returned module will have the same device and dtype as the original module.
    Random weights are initialized with the same mean and std as the original weights.

    :param module: (nn.Module) Module to replace the input channels in.
    :param in_channels: New number of input channels.
    :return: nn.Module
    """
    if isinstance(module, nn.Conv2d):

        if in_channels % module.groups != 0:
            raise ValueError(
                f"Incompatible number of input channels ({in_channels}) with the number of groups ({module.groups})."
                f"The number of input channels must be divisible by the number of groups."
            )

        new_module = nn.Conv2d(
            in_channels,
            module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,  # Be cautious: if in_channels % groups != 0, it will raise an error.
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        torch.nn.init.normal_(new_module.weight, mean=module.weight.mean().item(), std=module.weight.std().item())
        if module.bias is not None:
            torch.nn.init.normal_(new_module.bias, mean=module.bias.mean().item(), std=module.bias.std().item())

        return new_module
    else:
        raise ValueError(f"Module {module} does not support replacing the input channels")
