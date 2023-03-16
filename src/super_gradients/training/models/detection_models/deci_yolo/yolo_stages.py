from typing import Type

import torch
from torch import nn

from super_gradients.modules import Residual
from super_gradients.training.models.detection_models.csp_darknet53 import Conv


class CustomBlockBottleneck(nn.Module):
    def __init__(self, input_channels, output_channels, block_type: Type[nn.Module], activation_type: Type[nn.Module], shortcut: bool, use_alpha: bool):
        super().__init__()

        self.cv1 = block_type(input_channels, output_channels, activation_type=activation_type)
        self.cv2 = block_type(output_channels, output_channels, activation_type=activation_type)
        self.add = shortcut and input_channels == output_channels
        self.shortcut = Residual() if self.add else None
        if use_alpha:
            self.alpha = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        else:
            self.alpha = 1.0

    def forward(self, x):
        return self.alpha * self.shortcut(x) + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SequentialWithIntermediates(nn.Sequential):
    def __init__(self, output_intermediates, *args):
        super(SequentialWithIntermediates, self).__init__(*args)
        self.output_intermediates = output_intermediates

    def forward(self, input):
        if self.output_intermediates:
            output = [input]
            for module in self:
                output.append(module(output[-1]))
            return output
        #  For uniformity, we return a list even if we don't output intermediates
        return [super(SequentialWithIntermediates, self).forward(input)]


class DeciYOLOCSPLayer(nn.Module):
    # TODO: Maybe merge with CSPLayer
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int,
        block_type: Type[nn.Module],
        activation_type: Type[nn.Module],
        shortcut: bool = True,
        use_alpha: bool = True,
        expansion: float = 0.5,
        hidden_channels: int = None,
        concat_intermediates: bool = False,
    ):
        super(DeciYOLOCSPLayer, self).__init__()
        if hidden_channels is None:
            hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, stride=1, activation_type=activation_type)
        self.conv2 = Conv(in_channels, hidden_channels, 1, stride=1, activation_type=activation_type)
        self.conv3 = Conv(hidden_channels * (2 + concat_intermediates * num_bottlenecks), out_channels, 1, stride=1, activation_type=activation_type)
        module_list = [
            CustomBlockBottleneck(hidden_channels, hidden_channels, block_type, activation_type, shortcut, use_alpha) for _ in range(num_bottlenecks)
        ]
        self.bottlenecks = SequentialWithIntermediates(concat_intermediates, *module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        x = torch.cat((*x_1, x_2), dim=1)
        return self.conv3(x)
