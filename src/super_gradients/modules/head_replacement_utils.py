from typing import Union

import torch
from torch import nn

__all__ = ["replace_num_classes_with_random_weights"]


def replace_num_classes_with_random_weights(module: Union[nn.Conv2d, nn.Linear, nn.Module], num_classes: int) -> nn.Module:
    """
    Replace the number of classes in the module with random weights.
    This is useful for replacing the output layer of a detection/classification head.
    This implementation support Conv2d and Linear layers.
    Returned module will have the same device and dtype as the original module.
    Random weights are initialized with the same mean and std as the original weights.

    :param module: (nn.Module) Module to replace the number of classes in.
    :param num_classes: New number of classes.
    :return: nn.Module
    """
    if isinstance(module, nn.Conv2d):
        new_module = nn.Conv2d(
            module.in_channels,
            num_classes,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        torch.nn.init.normal_(new_module.weight, mean=module.weight.mean().item(), std=module.weight.std(dim=(0, 1, 2, 3)).item())
        if module.bias is not None:
            torch.nn.init.normal_(new_module.bias, mean=module.bias.mean().item(), std=module.bias.std(dim=0).item())

        return new_module
    elif isinstance(module, nn.Linear):
        new_module = nn.Linear(module.in_features, num_classes, device=module.weight.device, dtype=module.weight.dtype, bias=module.bias is not None)
        torch.nn.init.normal_(new_module.weight, mean=module.weight.mean().item(), std=module.weight.std(dim=(0, 1, 2)).item())
        if module.bias is not None:
            torch.nn.init.normal_(new_module.bias, mean=module.bias.mean().item(), std=module.bias.std(dim=0).item())

        return new_module
    else:
        raise ValueError(f"Module {module} does not support replacing the number of classes")
