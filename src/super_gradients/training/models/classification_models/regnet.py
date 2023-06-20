"""
Regnet - from paper: Designing Network Design Spaces - https://arxiv.org/pdf/2003.13678.pdf
Implementation of paradigm described in paper published by Facebook AI Research (FAIR)
@author: Signatrix GmbH
Code taken from: https://github.com/signatrix/regnet - MIT Licence
"""
import numpy as np
import torch
import torch.nn as nn
from math import sqrt

from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.modules import Residual
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.utils.regularization_utils import DropPath
from super_gradients.training.utils.utils import get_param


class Head(nn.Module):  # From figure 3
    def __init__(self, num_channels, num_classes, dropout_prob):
        super(Head, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Stem(nn.Module):  # From figure 3
    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.rl(x)
        return x


class XBlock(nn.Module):  # From figure 4
    def __init__(self, in_channels, out_channels, bottleneck_ratio, group_width, stride, se_ratio=None, droppath_prob=0.0):
        super(XBlock, self).__init__()
        inter_channels = int(out_channels // bottleneck_ratio)
        groups = int(inter_channels // group_width)

        self.conv_block_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU())
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
        )

        if se_ratio is not None:
            se_channels = in_channels // se_ratio
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Conv2d(inter_channels, se_channels, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(se_channels, inter_channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
                Residual(),
            )
            self.se_residual = Residual()
        else:
            self.se = None

        self.conv_block_3 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels), Residual()
            )
        else:
            self.shortcut = Residual()
        self.drop_path = DropPath(drop_prob=droppath_prob)
        self.rl = nn.ReLU()

    def forward(self, x):
        x1 = self.conv_block_1(x)
        x1 = self.conv_block_2(x1)
        if self.se is not None:
            x1 = self.se_residual(x1) * self.se(x1)

        x1 = self.conv_block_3(x1)
        x2 = self.shortcut(x)

        x1 = self.drop_path(x1)
        x = self.rl(x1 + x2)
        return x


class Stage(nn.Module):  # From figure 3
    def __init__(self, num_blocks, in_channels, out_channels, bottleneck_ratio, group_width, stride, se_ratio, droppath_prob):
        super(Stage, self).__init__()
        self.blocks = nn.Sequential()
        self.blocks.add_module("block_0", XBlock(in_channels, out_channels, bottleneck_ratio, group_width, stride, se_ratio, droppath_prob))
        for i in range(1, num_blocks):
            self.blocks.add_module("block_{}".format(i), XBlock(out_channels, out_channels, bottleneck_ratio, group_width, 1, se_ratio, droppath_prob))

    def forward(self, x):
        x = self.blocks(x)
        return x


class AnyNetX(SgModule):
    def __init__(
        self,
        ls_num_blocks,
        ls_block_width,
        ls_bottleneck_ratio,
        ls_group_width,
        stride,
        num_classes,
        se_ratio,
        backbone_mode,
        dropout_prob=0.0,
        droppath_prob=0.0,
        input_channels=3,
    ):
        super(AnyNetX, self).__init__()
        verify_correctness_of_parameters(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width)
        self.net = nn.Sequential()
        self.backbone_mode = backbone_mode
        prev_block_width = 32
        self.net.add_module("stem", Stem(in_channels=input_channels, out_channels=prev_block_width))

        for i, (num_blocks, block_width, bottleneck_ratio, group_width) in enumerate(zip(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width)):
            self.net.add_module(
                "stage_{}".format(i), Stage(num_blocks, prev_block_width, block_width, bottleneck_ratio, group_width, stride, se_ratio, droppath_prob)
            )
            prev_block_width = block_width
        # FOR BACK BONE MODE - DO NOT ADD THE HEAD (AVG_POOL + FC)
        if not self.backbone_mode:
            self.net.add_module("head", Head(ls_block_width[-1], num_classes, dropout_prob))
        self.initialize_weight()

        self.ls_block_width = ls_block_width
        self.dropout_prob = dropout_prob

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.net(x)
        return x

    def replace_head(self, new_num_classes=None, new_head=None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_head is not None:
            self.net.head = new_head
        else:
            self.net.head = Head(self.ls_block_width[-1], new_num_classes, self.dropout_prob)


def regnet_params_to_blocks(initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width):
    # We need to derive block width and number of blocks from initial parameters.
    parameterized_width = initial_width + slope * np.arange(network_depth)  # From equation 2
    parameterized_block = np.log(parameterized_width / initial_width) / np.log(quantized_param)  # From equation 3
    parameterized_block = np.round(parameterized_block)
    quantized_width = initial_width * np.power(quantized_param, parameterized_block)
    # We need to convert quantized_width to make sure that it is divisible by 8
    quantized_width = 8 * np.round(quantized_width / 8)
    ls_block_width, ls_num_blocks = np.unique(quantized_width.astype(np.int), return_counts=True)
    # At this points, for each stage, the above-calculated block width could be incompatible to group width
    # due to bottleneck ratio. Hence, we need to adjust the formers.
    # Group width could be swapped to number of groups, since their multiplication is block width
    ls_group_width = np.array([min(group_width, block_width // bottleneck_ratio) for block_width in ls_block_width])
    ls_block_width = (np.round(ls_block_width // bottleneck_ratio / group_width) * group_width).astype(np.int).tolist()
    ls_bottleneck_ratio = [bottleneck_ratio for _ in range(len(ls_block_width))]
    return ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width.tolist()


class RegNetX(AnyNetX):
    def __init__(
        self, initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width, stride, arch_params, se_ratio=None, input_channels=3
    ):
        ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width = regnet_params_to_blocks(
            initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width
        )

        # GET THE BACKBONE MODE FROM arch_params IF EXISTS - O.W. - SET AS FALSE
        backbone_mode = get_param(arch_params, "backbone_mode", False)
        dropout_prob = get_param(arch_params, "dropout_prob", 0.0)
        droppath_prob = get_param(arch_params, "droppath_prob", 0.0)
        super(RegNetX, self).__init__(
            ls_num_blocks,
            ls_block_width,
            ls_bottleneck_ratio,
            ls_group_width,
            stride,
            arch_params.num_classes,
            se_ratio,
            backbone_mode,
            dropout_prob,
            droppath_prob,
            input_channels,
        )


class RegNetY(RegNetX):
    # RegNetY = RegNetX + SE
    def __init__(self, initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width, stride, arch_params, se_ratio, input_channels=3):
        super(RegNetY, self).__init__(
            initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width, stride, arch_params, se_ratio, input_channels
        )


def verify_correctness_of_parameters(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width):
    """VERIFY THAT THE GIVEN PARAMETERS FIT THE SEARCH SPACE DEFINED IN THE REGNET PAPER"""
    err_message = "Parameters don't fit"
    assert len(set(ls_bottleneck_ratio)) == 1, f"{err_message} AnyNetXb"
    assert len(set(ls_group_width)) == 1, f"{err_message} AnyNetXc"
    assert all(i <= j for i, j in zip(ls_block_width, ls_block_width[1:])) is True, f"{err_message} AnyNetXd"
    if len(ls_num_blocks) > 2:
        assert all(i <= j for i, j in zip(ls_num_blocks[:-2], ls_num_blocks[1:-1])) is True, f"{err_message} AnyNetXe"
    # For each stage & each layer, number of channels (block width / bottleneck ratio) must be divisible by group width
    for block_width, bottleneck_ratio, group_width in zip(ls_block_width, ls_bottleneck_ratio, ls_group_width):
        assert int(block_width // bottleneck_ratio) % group_width == 0


@register_model(Models.CUSTOM_REGNET)
class CustomRegNet(RegNetX):
    def __init__(self, arch_params):
        """All parameters must be provided in arch_params other than SE"""
        super().__init__(
            initial_width=arch_params.initial_width,
            slope=arch_params.slope,
            quantized_param=arch_params.quantized_param,
            network_depth=arch_params.network_depth,
            bottleneck_ratio=arch_params.bottleneck_ratio,
            group_width=arch_params.group_width,
            stride=arch_params.stride,
            arch_params=arch_params,
            se_ratio=arch_params.se_ratio if hasattr(arch_params, "se_ratio") else None,
            input_channels=get_param(arch_params, "input_channels", 3),
        )


@register_model(Models.CUSTOM_ANYNET)
class CustomAnyNet(AnyNetX):
    def __init__(self, arch_params):
        """All parameters must be provided in arch_params other than SE"""
        super().__init__(
            ls_num_blocks=arch_params.ls_num_blocks,
            ls_block_width=arch_params.ls_block_width,
            ls_bottleneck_ratio=arch_params.ls_bottleneck_ratio,
            ls_group_width=arch_params.ls_group_width,
            stride=arch_params.stride,
            num_classes=arch_params.num_classes,
            se_ratio=arch_params.se_ratio if hasattr(arch_params, "se_ratio") else None,
            backbone_mode=get_param(arch_params, "backbone_mode", False),
            dropout_prob=get_param(arch_params, "dropout_prob", 0),
            droppath_prob=get_param(arch_params, "droppath_prob", 0),
            input_channels=get_param(arch_params, "input_channels", 3),
        )


@register_model(Models.NAS_REGNET)
class NASRegNet(RegNetX):
    def __init__(self, arch_params):
        """All parameters are provided as a single structure list: arch_params.structure"""
        structure = arch_params.structure
        super().__init__(
            initial_width=structure[0],
            slope=structure[1],
            quantized_param=structure[2],
            network_depth=structure[3],
            bottleneck_ratio=structure[4],
            group_width=structure[5],
            stride=structure[6],
            se_ratio=structure[7] if structure[7] > 0 else None,
            arch_params=arch_params,
        )


@register_model(Models.REGNETY200)
class RegNetY200(RegNetY):
    def __init__(self, arch_params):
        super().__init__(24, 36, 2.5, 13, 1, 8, 2, arch_params, 4)


@register_model(Models.REGNETY400)
class RegNetY400(RegNetY):
    def __init__(self, arch_params):
        super().__init__(48, 28, 2.1, 16, 1, 8, 2, arch_params, 4)


@register_model(Models.REGNETY600)
class RegNetY600(RegNetY):
    def __init__(self, arch_params):
        super().__init__(48, 33, 2.3, 15, 1, 16, 2, arch_params, 4)


@register_model(Models.REGNETY800)
class RegNetY800(RegNetY):
    def __init__(self, arch_params):
        super().__init__(56, 39, 2.4, 14, 1, 16, 2, arch_params, 4)
