from typing import Type, List, Tuple, Union, Dict
from abc import ABC, abstractmethod

import torch.nn as nn
import torch
import torch.nn.functional as F

from super_gradients.modules import ConvBNReLU
from super_gradients.modules.sampling import UpsampleMode
from super_gradients.common.object_names import ContextModules


class AbstractContextModule(nn.Module, ABC):
    @abstractmethod
    def output_channels(self):
        raise NotImplementedError


class SPPM(AbstractContextModule):
    """
    Simple Pyramid Pooling context Module.
    """

    def __init__(
        self,
        in_channels: int,
        inter_channels: int,
        out_channels: int,
        pool_sizes: List[Union[int, Tuple[int, int]]],
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.BILINEAR,
        align_corners: bool = False,
    ):
        """
        :param inter_channels: num channels in each pooling branch.
        :param out_channels: The number of output channels after pyramid pooling module.
        :param pool_sizes: spatial output sizes of the pooled feature maps.
        """
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    ConvBNReLU(in_channels, inter_channels, kernel_size=1, bias=False),
                )
                for pool_size in pool_sizes
            ]
        )
        self.conv_out = ConvBNReLU(inter_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners
        self.pool_sizes = pool_sizes

    def forward(self, x):
        out = None
        input_shape = x.shape[2:]
        for branch in self.branches:
            y = branch(x)
            y = F.interpolate(y, size=input_shape, mode=self.upsample_mode, align_corners=self.align_corners)
            out = y if out is None else out + y
        out = self.conv_out(out)
        return out

    def output_channels(self):
        return self.out_channels

    def prep_model_for_conversion(self, input_size: Union[tuple, list], stride_ratio: int = 32, **kwargs):
        """
        Replace Global average pooling with fixed kernels Average pooling, since dynamic kernel sizes are not supported
        when compiling to ONNX: `Unsupported: ONNX export of operator adaptive_avg_pool2d, input size not accessible.`
        """
        input_size = [x / stride_ratio for x in input_size[-2:]]
        for branch in self.branches:
            global_pool: nn.AdaptiveAvgPool2d = branch[0]
            # If not a global average pooling skip this. The module might be already converted to average pooling
            # modules.
            if not isinstance(global_pool, nn.AdaptiveAvgPool2d):
                continue
            out_size = global_pool.output_size
            out_size = out_size if isinstance(out_size, (tuple, list)) else (out_size, out_size)
            kernel_size = [int(i / o) for i, o in zip(input_size, out_size)]
            branch[0] = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)


class ASPP(AbstractContextModule):
    """
    ASPP bottleneck block. Splits the input to len(dilation_list) + 1, (a 1x1 conv) heads of differently dilated convolutions.
    The different heads will be concatenated and the output channel of each will be the
    input channel / len(dilation_list) + 1 so as to keep the same output channel as input channel.
    """

    def __init__(self, in_channels: int, dilation_list: List[int], in_out_ratio: float = 1.0, use_bias: bool = False, **kwargs):
        """
        :param dilation_list: list of dilation rates, the num of dilation branches should be set so that there is a
            whole division of the input channels, see assertion below.
        :param in_out_ratio: output / input num of channels ratio.
        :param use_bias: legacy parameter to support PascalVOC frontier checkpoints that were trained by mistake with
            extra redundant biases before batchnorm operators. should be set to `False` for new training processes.
        """
        super().__init__()
        num_dilation_branches = len(dilation_list) + 1
        inter_ratio = num_dilation_branches / in_out_ratio
        assert in_channels % inter_ratio == 0
        inter_channels = int(in_channels / inter_ratio)

        self.dilated_conv_list = nn.ModuleList(
            [
                ConvBNReLU(in_channels, inter_channels, kernel_size=1, dilation=1, bias=use_bias),
                *[ConvBNReLU(in_channels, inter_channels, kernel_size=3, dilation=d, padding=d, bias=use_bias) for d in dilation_list],
            ]
        )

        self.out_channels = inter_channels * num_dilation_branches

    def output_channels(self):
        return self.out_channels

    def forward(self, x):
        x = torch.cat([dilated_conv(x) for dilated_conv in self.dilated_conv_list], dim=1)
        return x


CONTEXT_TYPE_DICT: Dict[str, Type[AbstractContextModule]] = {ContextModules.ASPP: ASPP, ContextModules.SPPM: SPPM}
