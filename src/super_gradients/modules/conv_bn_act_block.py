from typing import Union, Tuple, Type

from torch import nn


class ConvBNAct(nn.Module):
    """
    Class for Convolution2d-Batchnorm2d-Activation layer.
        Default behaviour is Conv-BN-Act. To exclude Batchnorm module use
        `use_normalization=False`, to exclude activation use `activation_type=None`.
    For convolution arguments documentation see `nn.Conv2d`.
    For batchnorm arguments documentation see `nn.BatchNorm2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        activation_type: Type[nn.Module],
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        use_normalization: bool = True,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        activation_kwargs=None,
    ):

        super().__init__()
        if activation_kwargs is None:
            activation_kwargs = {}

        self.seq = nn.Sequential()
        self.seq.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            ),
        )

        if use_normalization:
            self.seq.add_module(
                "bn",
                nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype),
            )
        if activation_type is not None:
            self.seq.add_module("act", activation_type(**activation_kwargs))

    def forward(self, x):
        return self.seq(x)
