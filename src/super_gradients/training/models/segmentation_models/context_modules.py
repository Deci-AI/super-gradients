import torch.nn as nn
import torch
from typing import Type, List, MutableMapping
from super_gradients.training.utils.module_utils import ConvBNReLU
from abc import ABC, abstractmethod


class AbstractContextModule(ABC):
    @abstractmethod
    def output_channels(self):
        raise NotImplementedError


class ASPP(nn.Module, AbstractContextModule):
    """
    ASPP bottleneck block. Splits the input to len(dilation_list) + 1, (a 1x1 conv) heads of differently dilated convolutions.
    The different heads will be concatenated and the output channel of each will be the
    input channel / len(dilation_list) + 1 so as to keep the same output channel as input channel.
    """

    def __init__(self,
                 in_channels: int,
                 dilation_list: List[int],
                 in_out_ratio: float = 1.,
                 use_bias: bool = False,
                 **kwargs):
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

        self.dilated_conv_list = nn.ModuleList([
            ConvBNReLU(in_channels, inter_channels, kernel_size=1, dilation=1, bias=use_bias),
            *[ConvBNReLU(in_channels, inter_channels, kernel_size=3, dilation=d, padding=d, bias=use_bias)
              for d in dilation_list]
        ])

        self.out_channels = inter_channels * num_dilation_branches

    def output_channels(self):
        return self.out_channels

    def forward(self, x):
        x = torch.cat([dilated_conv(x) for dilated_conv in self.dilated_conv_list], dim=1)
        return x


CONTEXT_TYPE_DICT: MutableMapping[str, Type[AbstractContextModule]] = {
    "aspp": ASPP,
}


def build_context_module(context_module_name: str, context_module_params: dict, in_channels: int):
    if context_module_name not in CONTEXT_TYPE_DICT.keys():
        raise NotImplementedError(f"Context module type: '{context_module_name}' is not implemented.")
    return CONTEXT_TYPE_DICT[context_module_name](in_channels=in_channels, **context_module_params)
