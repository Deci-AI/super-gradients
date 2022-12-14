from typing import Union, Tuple

from torch import nn

from super_gradients.modules.conv_bn_act_block import ConvBNAct


class ConvBNReLU(ConvBNAct):
    """
    Class for Convolution2d-Batchnorm2d-Relu layer. To exclude Batchnorm module use
        `use_normalization=False`, to exclude Relu activation use `use_activation=False`.

    It exists to keep backward compatibility and will be superseeded by ConvBNAct in future releases.
    For new classes please use ConvBNAct instead.

    For convolution arguments documentation see `nn.Conv2d`.
    For batchnorm arguments documentation see `nn.BatchNorm2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
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
        use_activation: bool = True,
        inplace: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            activation_type=nn.ReLU if use_activation else None,
            activation_kwargs=dict(inplace=inplace) if inplace else None,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            use_normalization=use_normalization,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
