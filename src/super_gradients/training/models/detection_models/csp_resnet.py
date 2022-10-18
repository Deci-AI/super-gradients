import collections
from typing import List, Union, Type

import torch
from torch import nn, Tensor

__all__ = ["CSPResNet"]


class ConvBNLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size: int, stride: int, padding: int, activation_type: Type[nn.Module]):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(ch_out)
        self.act = activation_type()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class RepVggBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, act: Type, alpha: bool = False):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, kernel_size=3, stride=1, padding=1, activation_type=nn.Identity)
        self.conv2 = ConvBNLayer(ch_in, ch_out, kernel_size=1, stride=1, padding=0, activation_type=nn.Identity)
        self.act = act()
        if alpha:
            self.alpha = torch.nn.Parameter(torch.tensor([1]), requires_grad=True)
        else:
            self.alpha = None

    def forward(self, x):
        if hasattr(self, "conv"):
            y = self.conv(x)
        else:
            if self.alpha:
                y = self.conv1(x) + self.alpha * self.conv2(x)
            else:
                y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        """
        Prepare the model to be converted to ONNX or other frameworks.
        Typically, this function will freeze the size of layers which is otherwise flexible, replace some modules
        with convertible substitutes and remove all auxiliary or training related parts.
        :param input_size: [H,W]
        """
        if self.training:
            raise RuntimeError("Module has to be in eval mode to be converted")

        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(
                in_channels=self.ch_in, out_channels=self.ch_out, kernel_size=3, stride=1, padding=1, groups=1
            )
        kernel, bias = self._get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__("conv1")
        self.__delattr__("conv2")

    def _get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + self.alpha * bias1x1
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, activation_type: Type[nn.Module], shortcut=True, use_alpha=False):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, kernel_size=3, stride=1, padding=1, activation_type=activation_type)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=activation_type, alpha=use_alpha)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class EffectiveSELayer(nn.Module):
    """Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels: int):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)


class CSPResStage(nn.Module):
    def __init__(
        self,
        block_fn,
        ch_in: int,
        ch_out: int,
        n,
        stride: int,
        activation_type: Type[nn.Module],
        use_attention: bool = True,
        use_alpha: bool = False,
    ):
        super(CSPResStage, self).__init__()

        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(ch_in, ch_mid, 3, stride=2, padding=1, activation_type=activation_type)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(
            ch_mid, ch_mid // 2, kernel_size=1, stride=1, padding=0, activation_type=activation_type
        )
        self.conv2 = ConvBNLayer(
            ch_mid, ch_mid // 2, kernel_size=1, stride=1, padding=0, activation_type=activation_type
        )
        self.blocks = nn.Sequential(
            *[
                block_fn(ch_mid // 2, ch_mid // 2, activation_type=activation_type, shortcut=True, use_alpha=use_alpha)
                for i in range(n)
            ]
        )
        if use_attention:
            self.attn = EffectiveSELayer(ch_mid)
        else:
            self.attn = nn.Identity()

        self.conv3 = ConvBNLayer(ch_mid, ch_out, kernel_size=1, stride=1, padding=0, activation_type=activation_type)

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
    def __init__(
        self,
        layers=[3, 6, 6, 3],
        channels=[64, 128, 256, 512, 1024],
        activation_type: Type[nn.Module] = torch.nn.SiLU,
        return_idx=[1, 2, 3],
        use_large_stem: bool = True,
        width_mult=1.0,
        depth_mult=1.0,
        use_alpha: bool = False,
    ):
        super().__init__()
        channels = [max(round(num_channels * width_mult), 1) for num_channels in channels]
        layers = [max(round(num_layers * depth_mult), 1) for num_layers in layers]

        if use_large_stem:
            self.stem = nn.Sequential(
                collections.OrderedDict(
                    [
                        (
                            "conv1",
                            ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, activation_type=activation_type),
                        ),
                        (
                            "conv2",
                            ConvBNLayer(
                                channels[0] // 2,
                                channels[0] // 2,
                                3,
                                stride=1,
                                padding=1,
                                activation_type=activation_type,
                            ),
                        ),
                        (
                            "conv3",
                            ConvBNLayer(
                                channels[0] // 2, channels[0], 3, stride=1, padding=1, activation_type=activation_type
                            ),
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
                            ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, activation_type=activation_type),
                        ),
                        (
                            "conv2",
                            ConvBNLayer(
                                channels[0] // 2, channels[0], 3, stride=1, padding=1, activation_type=activation_type
                            ),
                        ),
                    ]
                )
            )

        n = len(channels) - 1
        self.stages = nn.ModuleList(
            [
                CSPResStage(
                    BasicBlock,
                    channels[i],
                    channels[i + 1],
                    layers[i],
                    2,
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

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        """
        Prepare the model to be converted to ONNX or other frameworks.
        Typically, this function will freeze the size of layers which is otherwise flexible, replace some modules
        with convertible substitutes and remove all auxiliary or training related parts.
        :param input_size: [H,W]
        """
        for module in self.modules():
            if isinstance(module, RepVggBlock):
                module.prep_model_for_conversion(input_size)
