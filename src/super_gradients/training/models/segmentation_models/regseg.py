"""
Implementation of paper: "Rethink Dilated Convolution for Real-time Semantic Segmentation", https://arxiv.org/pdf/2111.09957.pdf
Based on original implementation: https://github.com/RolandGao/RegSeg, cloned 23/12/2021, commit c07a833
"""
from typing import List

import torch
import torch.nn as nn

from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.training.models import SgModule
from super_gradients.training.utils import HpmStruct, get_param
from super_gradients.modules import ConvBNReLU

DEFAULT_REGSEG48_BACKBONE_PARAMS = {
    "stages": [
        [[48, [1], 16, 2, 4]],
        [[128, [1], 16, 2, 4], *[[128, [1], 16, 1, 4]] * 2],
        [
            [256, [1], 16, 2, 4],
            [256, [1], 16, 1, 4],
            [256, [1, 2], 16, 1, 4],
            *[[256, [1, 4], 16, 1, 4]] * 4,
            *[[256, [1, 14], 16, 1, 4]] * 6,
            [320, [1, 14], 16, 1, 4],
        ],
    ]
}

DEFAULT_REGSEG53_BACKBONE_PARAMS = {
    "stages": [
        [[48, [1], 24, 2, 4], [48, [1], 24, 1, 4]],
        [[120, [1], 24, 2, 4], *[[120, [1], 24, 1, 4]] * 5],
        [
            [336, [1], 24, 2, 4],
            [336, [1], 24, 1, 4],
            [336, [1, 2], 24, 1, 4],
            *[[336, [1, 4], 24, 1, 4]] * 4,
            *[[336, [1, 14], 24, 1, 4]] * 6,
            [384, [1, 14], 24, 1, 4],
        ],
    ]
}

DEFAULT_REGSEG48_DECODER_PARAMS = {"projection_out_channels": [8, 128, 128], "interpolation": "bilinear"}

DEFAULT_REGSEG53_DECODER_PARAMS = {"projection_out_channels": [16, 256, 256], "interpolation": "bilinear"}

DEFAULT_REGSEG_HEAD_PARAMS = {"dropout": 0.0, "interpolation": "bilinear", "align_corners": False, "upsample_factor": 4}

DEFAULT_REGSEG48_HEAD_PARAMS = {"mid_channels": 64, **DEFAULT_REGSEG_HEAD_PARAMS}

DEFAULT_REGSEG53_HEAD_PARAMS = {"mid_channels": 128, **DEFAULT_REGSEG_HEAD_PARAMS}


class SqueezeAndExcitationBlock(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int):
        super().__init__()
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, bottleneck_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, in_channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.se_block(x)
        return x * y


class AdaptiveShortcutBlock(nn.Module):
    """
    Adaptive shortcut makes the following adaptations, if needed:
    Applying pooling if stride > 1
    Applying 1x1 conv if in/out channels are different or if pooling was applied
    If stride is 1 and in/out channels are the same, then the shortcut is just an identity
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        shortcut_layers = [nn.Identity()]
        if stride != 1:
            shortcut_layers[0] = nn.AvgPool2d(stride, stride, ceil_mode=True)  # override the identity layer
        if in_channels != out_channels or stride != 1:
            shortcut_layers.append(ConvBNReLU(in_channels, out_channels, kernel_size=1, bias=False, use_activation=False))
        self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        return self.shortcut(x)


class SplitDilatedGroupConvBlock(nn.Module):
    """
    Splits the input to "dilation groups", following grouped convolution with different dilation for each group
    """

    def __init__(self, in_channels: int, split_dilations: List[int], group_width_per_split: int, stride: int, bias: bool):
        """
        :param split_dilations:         a list specifying the required dilations.
                                        the input will be split into len(dilations) groups,
                                        group [i] will be convolved with grouped dilated (dilations[i]) convolution
        :param group_width_per_split:   the group width for the *inner* dilated convolution
        """
        super().__init__()
        self.num_splits = len(split_dilations)
        assert in_channels % self.num_splits == 0, f"Cannot split {in_channels} to {self.num_splits} groups with equal size."
        group_channels = in_channels // self.num_splits
        assert group_channels % group_width_per_split == 0, (
            f"Cannot split {group_channels} channels ({in_channels} / {self.num_splits} splits)" f" to groups with {group_width_per_split} channels per group."
        )
        inner_groups = group_channels // group_width_per_split
        self.convs = nn.ModuleList(
            nn.Conv2d(group_channels, group_channels, 3, padding=d, dilation=d, stride=stride, bias=bias, groups=inner_groups) for d in split_dilations
        )
        self._splits = [in_channels // self.num_splits] * self.num_splits

    def forward(self, x):
        x = torch.split(x, self._splits, dim=1)
        return torch.cat([self.convs[i](x[i]) for i in range(self.num_splits)], dim=1)


class DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations: List[int], group_width: int, stride: int, se_ratio: int = 4):
        """
        :param dilations:           a list specifying the required dilations.
                                    the input will be split into len(dilations) groups,
                                    group [i] will be convolved with grouped dilated (dilations[i]) convolution
        :param group_width:         the group width for the dilated convolution(s)
        :param se_ratio:            the ratio of the squeeze-and-excitation block w.r.t in_channels (as in the paper)
                                    for example: a value of 4 translates to in_channels // 4
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations
        self.group_width = group_width
        self.stride = stride
        self.se_ratio = se_ratio
        self.shortcut = AdaptiveShortcutBlock(in_channels, out_channels, stride)
        groups = out_channels // group_width

        if len(dilations) == 1:  # minor optimization: no need to split if we only have 1 dilation group
            dilation = dilations[0]
            dilated_conv = nn.Conv2d(out_channels, out_channels, 3, stride=stride, groups=groups, padding=dilation, dilation=dilation, bias=False)
        else:
            dilated_conv = SplitDilatedGroupConvBlock(out_channels, dilations, group_width_per_split=group_width, stride=stride, bias=False)

        self.d_block_path = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, kernel_size=1, bias=False),
            dilated_conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # the ratio of se block applied to `in_channels` as in the original paper
            SqueezeAndExcitationBlock(out_channels, in_channels // se_ratio),
            ConvBNReLU(out_channels, out_channels, 1, use_activation=False, bias=False),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.shortcut(x)
        x2 = self.d_block_path(x)
        out = self.relu(x1 + x2)
        return out

    def __str__(self):
        return (
            f"{self.__class__.__name__}_in{self.in_channels}_out{self.out_channels}" f"_d{self.dilations}_gw{self.group_width}_s{self.stride}_se{self.se_ratio}"
        )


class RegSegDecoder(nn.Module):
    """
    This implementation follows the paper. No 'pattern' in this decoder, so it is specific to 3 stages
    """

    def __init__(self, backbone_output_channels: List[int], decoder_config: dict):
        super().__init__()
        projection_out_channels = decoder_config["projection_out_channels"]

        assert len(backbone_output_channels) == len(projection_out_channels) == 3, "This decoder is specific for 3 stages"

        self.projections = nn.ModuleList(
            [ConvBNReLU(in_channels, out_channels, 1, bias=False) for in_channels, out_channels in zip(backbone_output_channels, projection_out_channels)]
        )
        self.upsample = nn.Upsample(scale_factor=2, mode=decoder_config["interpolation"], align_corners=True)
        mid_channels = projection_out_channels[1]
        self.conv_bn_relu = ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels // 2, kernel_size=3, padding=1, bias=False)
        self.out_channels = mid_channels // 2 + projection_out_channels[0]  # original implementation: concat

    def forward(self, x_stages):
        proj2 = self.projections[2](x_stages[2])
        proj2 = self.upsample(proj2)
        proj1 = self.projections[1](x_stages[1])
        proj1 = proj1 + proj2
        proj1 = self.conv_bn_relu(proj1)
        proj1 = self.upsample(proj1)
        proj0 = self.projections[0](x_stages[0])
        proj0 = torch.cat((proj1, proj0), dim=1)
        return proj0


class RegSegHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, head_config: dict):
        super().__init__()
        layers = list()
        layers.append(ConvBNReLU(in_channels, head_config["mid_channels"], 3, bias=False, padding=1))
        if head_config["dropout"] > 0:
            layers.append(nn.Dropout(head_config["dropout"], inplace=False))

        layers.append(nn.Conv2d(head_config["mid_channels"], num_classes, 1))
        layers.append(nn.Upsample(scale_factor=head_config["upsample_factor"], mode=head_config["interpolation"], align_corners=head_config["align_corners"]))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


class RegSegBackbone(nn.Module):
    def __init__(self, in_channels: int, backbone_config: dict):
        super().__init__()
        self.stages, self.backbone_output_channels = self._generate_stages(in_channels, backbone_config["stages"])

    @staticmethod
    def _generate_stages(in_channels, backbone_stages):
        prev_out_channels = in_channels
        backbone_channels = list()
        stages = nn.ModuleList()
        for stage in backbone_stages:
            stage_blocks = nn.Sequential()
            for i, (out_channels, dilations, group_width, stride, se_ratio) in enumerate(stage):
                d_block = DBlock(prev_out_channels, out_channels, dilations, group_width, stride, se_ratio)
                prev_out_channels = d_block.out_channels
                stage_blocks.add_module(f"{str(d_block)}#{i}", d_block)  # NOTE: {i} distinguishes blocks with same name
            stages.append(stage_blocks)
            backbone_channels.append(prev_out_channels)
        return stages, backbone_channels

    def forward(self, x):
        outputs = list()
        x_in = x
        for stage in self.stages:
            x_out = stage(x_in)
            outputs.append(x_out)
            x_in = x_out  # last stage out is next stage in
        return outputs

    def get_backbone_output_number_of_channels(self):
        return self.backbone_output_channels


class RegSeg(SgModule):
    def __init__(self, stem, backbone, decoder, head):
        super().__init__()
        self.stem = stem
        self.backbone = backbone
        self.decoder = decoder
        self.head = head

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.decoder(x)
        x = self.head(x)
        return x

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        multiply_head_lr = get_param(training_params, "multiply_head_lr", 1)
        multiply_lr_params, no_multiply_params = {}, {}
        for name, param in self.named_parameters():
            if "head." in name:
                multiply_lr_params[name] = param
            else:
                no_multiply_params[name] = param

        multiply_lr_params, no_multiply_params = multiply_lr_params.items(), no_multiply_params.items()

        param_groups = [
            {"named_params": no_multiply_params, "lr": lr, "name": "no_multiply_params"},
            {"named_params": multiply_lr_params, "lr": lr * multiply_head_lr, "name": "multiply_lr_params"},
        ]
        return param_groups

    def update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int, training_params: HpmStruct, total_batch: int) -> list:
        multiply_head_lr = get_param(training_params, "multiply_head_lr", 1)
        for param_group in param_groups:
            param_group["lr"] = lr
            if param_group["name"] == "multiply_lr_params":
                param_group["lr"] *= multiply_head_lr
        return param_groups

    def replace_head(self, new_num_classes: int, head_config: dict):
        self.head = RegSegHead(self.decoder.out_channels, new_num_classes, head_config)


@register_model(Models.REGSEG48)
class RegSeg48(RegSeg):
    def __init__(self, arch_params: HpmStruct):
        num_classes = get_param(arch_params, "num_classes")
        stem = ConvBNReLU(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        backbone = RegSegBackbone(in_channels=32, backbone_config=DEFAULT_REGSEG48_BACKBONE_PARAMS)
        decoder = RegSegDecoder(backbone.get_backbone_output_number_of_channels(), DEFAULT_REGSEG48_DECODER_PARAMS)
        head = RegSegHead(decoder.out_channels, num_classes, DEFAULT_REGSEG48_HEAD_PARAMS)
        super().__init__(stem, backbone, decoder, head)

    def replace_head(self, new_num_classes: int, head_config: dict = None):
        head_config = head_config or DEFAULT_REGSEG48_HEAD_PARAMS
        super().replace_head(new_num_classes, head_config)


class RegSeg53(RegSeg):
    def __init__(self, arch_params: HpmStruct):
        num_classes = get_param(arch_params, "num_classes")
        stem = ConvBNReLU(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        backbone = RegSegBackbone(in_channels=32, backbone_config=DEFAULT_REGSEG53_BACKBONE_PARAMS)
        decoder = RegSegDecoder(backbone.get_backbone_output_number_of_channels(), DEFAULT_REGSEG53_DECODER_PARAMS)
        head = RegSegHead(decoder.out_channels, num_classes, DEFAULT_REGSEG53_HEAD_PARAMS)
        super().__init__(stem, backbone, decoder, head)

    def replace_head(self, new_num_classes: int, head_config: dict = None):
        head_config = head_config or DEFAULT_REGSEG53_HEAD_PARAMS
        super().replace_head(new_num_classes, head_config)
