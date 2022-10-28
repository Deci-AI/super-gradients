import collections
from typing import List, Type

import torch
from torch import nn, Tensor

from super_gradients.modules import RepVGGBlock, EffectiveSEBlock
from super_gradients.training.utils.module_utils import ConvBNAct

__all__ = ["CSPResNet"]


class CSPResNetBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation_type: Type[nn.Module], shortcut=True, use_alpha=False):
        super().__init__()
        if shortcut and in_channels != out_channels:
            raise RuntimeError("Number of input channels must be equal to the number of output channels when shortcut=True")
        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation_type=activation_type, bias=False)
        self.conv2 = RepVGGBlock(
            out_channels, out_channels, activation_type=activation_type, se_type=nn.Identity, use_residual_connection=False, use_alpha=use_alpha
        )
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class CSPResStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks,
        stride: int,
        activation_type: Type[nn.Module],
        use_attention: bool = True,
        use_alpha: bool = False,
    ):
        """

        :param block_type:
        :param in_channels:
        :param out_channels:
        :param num_blocks:
        :param stride: Desired downsampling for the stage (Usually 2)
        :param activation_type:
        :param use_attention:
        :param use_alpha: If True, enables additional learnable weighting parameter for 1x1 branch in underlying RepVGG blocks (PP-Yolo-E Plus)
        """
        super().__init__()

        mid_channels = (in_channels + out_channels) // 2
        if stride != 1:
            self.conv_down = ConvBNAct(in_channels, mid_channels, 3, stride=stride, padding=1, activation_type=activation_type, bias=False)
        else:
            self.conv_down = None
        self.conv1 = ConvBNAct(mid_channels, mid_channels // 2, kernel_size=1, stride=1, padding=0, activation_type=activation_type, bias=False)
        self.conv2 = ConvBNAct(mid_channels, mid_channels // 2, kernel_size=1, stride=1, padding=0, activation_type=activation_type, bias=False)
        self.blocks = nn.Sequential(
            *[
                CSPResNetBasicBlock(
                    in_channels=mid_channels // 2,
                    out_channels=mid_channels // 2,
                    activation_type=activation_type,
                    shortcut=True,
                    use_alpha=use_alpha,
                )
                for _ in range(num_blocks)
            ]
        )
        if use_attention:
            self.attn = EffectiveSEBlock(mid_channels)
        else:
            self.attn = nn.Identity()

        self.conv3 = ConvBNAct(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, activation_type=activation_type, bias=False)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.concat([y1, y2], dim=1)
        y = self.attn(y)
        y = self.conv3(y)
        return y


class CSPResNet(nn.Module):
    """
    CSPResNet backbone
    """

    def __init__(
        self,
        layers=[3, 6, 6, 3],
        channels=[64, 128, 256, 512, 1024],
        activation_type: Type[nn.Module] = torch.nn.SiLU,
        return_idx=[1, 2, 3],
        use_large_stem: bool = True,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        use_alpha: bool = False,
    ):
        """

        :param layers:
        :param channels:
        :param activation_type:
        :param return_idx:
        :param use_large_stem:
        :param width_mult:
        :param depth_mult:
        :param use_alpha:
        """
        super().__init__()
        channels = [max(round(num_channels * width_mult), 1) for num_channels in channels]
        layers = [max(round(num_layers * depth_mult), 1) for num_layers in layers]

        if use_large_stem:
            self.stem = nn.Sequential(
                collections.OrderedDict(
                    [
                        (
                            "conv1",
                            ConvBNAct(3, channels[0] // 2, 3, stride=2, padding=1, activation_type=activation_type, bias=False),
                        ),
                        (
                            "conv2",
                            ConvBNAct(
                                channels[0] // 2,
                                channels[0] // 2,
                                3,
                                stride=1,
                                padding=1,
                                activation_type=activation_type,
                                bias=False,
                            ),
                        ),
                        (
                            "conv3",
                            ConvBNAct(channels[0] // 2, channels[0], 3, stride=1, padding=1, activation_type=activation_type, bias=False),
                        ),
                    ]
                )
            )
        else:
            self.stem = nn.Sequential(
                collections.OrderedDict(
                    [
                        (
                            "conv1",
                            ConvBNAct(3, channels[0] // 2, 3, stride=2, padding=1, activation_type=activation_type, bias=False),
                        ),
                        (
                            "conv2",
                            ConvBNAct(channels[0] // 2, channels[0], 3, stride=1, padding=1, activation_type=activation_type, bias=False),
                        ),
                    ]
                )
            )

        n = len(channels) - 1
        self.stages = nn.ModuleList(
            [
                CSPResStage(
                    channels[i],
                    channels[i + 1],
                    layers[i],
                    stride=2,
                    activation_type=activation_type,
                    use_alpha=use_alpha,
                )
                for i in range(n)
            ]
        )

        self._out_channels = channels[1:]
        self._out_strides = [4 * 2**i for i in range(n)]
        self.return_idx = return_idx

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)

        return outs
