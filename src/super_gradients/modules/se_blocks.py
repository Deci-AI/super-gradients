import torch
from torch import nn, Tensor
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Spatial Squeeze and Channel Excitation Block (cSE).

    Figure 1, Variant a from https://arxiv.org/abs/1808.08127v1
    """

    def __init__(self, in_channels: int, internal_neurons: int):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=in_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=in_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = in_channels

    def forward(self, inputs: Tensor) -> Tensor:
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class EffectiveSEBlock(nn.Module):
    """Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.project = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.project(x_se)
        return x * self.act(x_se)
