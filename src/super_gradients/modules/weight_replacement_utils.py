from typing import Optional, Callable

import torch
from torch import nn

__all__ = ["replace_conv2d_input_channels", "replace_conv2d_input_channels_with_random_weights"]


def replace_conv2d_input_channels(conv: nn.Conv2d, in_channels: int, fn: Optional[Callable[[nn.Conv2d, int], nn.Conv2d]] = None) -> nn.Module:
    """Instantiate a new Conv2d module with same attributes as input Conv2d, except for the input channels.

    :param conv:        Conv2d to replace the input channels in.
    :param in_channels: New number of input channels.
    :param fn:          (Optional) Function to instantiate the new Conv2d.
                        By default, it will initialize the new weights with the same mean and std as the original weights.
    :return:            Conv2d with new number of input channels.
    """
    if fn:
        return fn(conv, in_channels)
    else:
        return replace_conv2d_input_channels_with_random_weights(conv=conv, in_channels=in_channels)


def replace_conv2d_input_channels_with_random_weights(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """
    Replace the input channels in the input Conv2d with random weights.
    Returned module will have the same device and dtype as the original module.
    Random weights are initialized with the same mean and std as the original weights.

    :param conv:        Conv2d to replace the input channels in.
    :param in_channels: New number of input channels.
    :return:            Conv2d with new number of input channels.
    """

    if in_channels % conv.groups != 0:
        raise ValueError(
            f"Incompatible number of input channels ({in_channels}) with the number of groups ({conv.groups})."
            f"The number of input channels must be divisible by the number of groups."
        )

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )

    if in_channels <= conv.in_channels:
        new_conv.weight.data = conv.weight.data[:, :in_channels, ...]
    else:
        new_conv.weight.data[:, : conv.in_channels, ...] = conv.weight.data

        # Pad the remaining channels with random weights
        torch.nn.init.normal_(new_conv.weight.data[:, conv.in_channels :, ...], mean=conv.weight.mean().item(), std=conv.weight.std().item())

    if conv.bias is not None:
        torch.nn.init.normal_(new_conv.bias, mean=conv.bias.mean().item(), std=conv.bias.std().item())

    return new_conv
