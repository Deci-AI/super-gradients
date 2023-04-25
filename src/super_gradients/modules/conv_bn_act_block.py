from typing import Union, Tuple, Type

from torch import nn

from super_gradients.modules.utils import autopad


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


class Conv(nn.Module):
    # STANDARD CONVOLUTION
    # TODO: This class is illegaly similar to ConvBNAct, and the only reason it exists is due to fact that some models were using it
    # previosly and one have to find a bulletproof way drop this class but still be able to load models that were using it. Perhaps
    # it is possible through load_state_dict / save_state_dict magic.
    def __init__(self, input_channels, output_channels, kernel, stride, activation_type: Type[nn.Module], padding: int = None, groups: int = None):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel, stride, autopad(kernel, padding), groups=groups or 1, bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = activation_type()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
