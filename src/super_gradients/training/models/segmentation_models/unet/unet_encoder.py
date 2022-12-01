import math
from typing import List, Type, Optional
from enum import Enum
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from super_gradients.training.models.classification_models.regnet import XBlock
from super_gradients.training.models.classification_models.repvgg import RepVGGBlock
from super_gradients.training.models.segmentation_models.stdc import STDCBlock
from super_gradients.training.models import SgModule, HpmStruct
from super_gradients.modules import ConvBNReLU
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.factories.type_factory import TypeFactory


class AntiAliasDownsample(nn.Module):
    def __init__(self, in_channels: int, stride: int):
        super().__init__()
        self.kernel_size = 3
        self.stride = stride
        self.channels = in_channels

        a = torch.tensor([1.0, 2.0, 1.0])

        filt = a[:, None] * a[None, :]
        filt = filt / torch.sum(filt)

        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, x):
        return F.conv2d(x, self.filt, stride=self.stride, padding=1, groups=self.channels)


class AbstractUNetBackbone(nn.Module, ABC):
    """
    All backbones for UNet segmentation models must implement this class.
    """

    @abstractmethod
    def get_backbone_output_number_of_channels(self) -> List[int]:
        """
        :return: list of stages num channels.
        """
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        :return: list of skip features from different resolutions to be fused by the decoder.
        """
        raise NotImplementedError()


class BackboneStage(nn.Module, ABC):
    """
    BackboneStage abstract class to define a stage in UnetBackbone. Each stage include blocks which their amounts is
    defined by `num_blocks`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int, num_blocks: int, **kwargs):
        super().__init__()
        self.blocks = self.build_stage(in_channels, out_channels, stride=stride, num_blocks=num_blocks, **kwargs)

    @abstractmethod
    def build_stage(self, in_channels: int, out_channels: int, stride: int, num_blocks: int, **kwargs) -> nn.Sequential:
        raise NotImplementedError()

    def forward(self, x):
        return self.blocks(x)


class STDCStage(BackboneStage):
    """
    STDC stage with STDCBlock as building block.
    """

    def build_stage(self, in_channels: int, out_channels: int, stride: int, num_blocks: int, steps: int, stdc_downsample_mode: str, **kwargs):
        """
        :param steps: The total number of convs in this module, 1 conv 1x1 and (steps - 1) conv3x3.
        :param stdc_downsample_mode: downsample mode in stdc block, supported `avg_pool` for average-pooling and
         `dw_conv` for depthwise-convolution.
        :return:
        """
        self.assert_divisible_channels(out_channels, steps)
        blocks = []
        # STDC blocks
        blocks.extend(
            [
                STDCBlock(in_channels, out_channels, stride=stride, steps=steps, stdc_downsample_mode=stdc_downsample_mode),
                *[STDCBlock(out_channels, out_channels, stride=1, steps=steps, stdc_downsample_mode=stdc_downsample_mode) for _ in range(num_blocks - 1)],
            ]
        )
        return nn.Sequential(*blocks)

    @staticmethod
    def assert_divisible_channels(num_channels: int, steps: int):
        """
        STDC block refactors the convolution operator by applying several smaller convolution with num of filters that
        decrease w.r.t the num of steps. The ratio to the smallest num of channels is `2 ** (steps - 1)`,
        thus this method assert that the stage num of channels is divisible by the above ratio.
        """
        channels_ratio = 2 ** (steps - 1)
        if num_channels % channels_ratio != 0:
            raise AssertionError(
                f"Num channels: {num_channels}, isn't divisible by the channels width ratio:"
                f" {channels_ratio}, when initiating an STDC block with steps: {steps}"
            )


class RepVGGStage(BackboneStage):
    """
    RepVGG stage with RepVGGBlock as building block. If `anti_alias=True`, `AntiAliasDownsample` module is used for
    downsampling.
    """

    def build_stage(self, in_channels: int, out_channels: int, stride: int, num_blocks: int, anti_alias: bool, **kwargs):
        blocks = []
        # Anti alias gaussian down-sampling
        if anti_alias and stride == 2:
            blocks.append(AntiAliasDownsample(in_channels, stride))
            stride = 1
        # RepVGG blocks
        blocks.extend([RepVGGBlock(in_channels, out_channels, stride=stride), *[RepVGGBlock(out_channels, out_channels) for _ in range(num_blocks - 1)]])
        return nn.Sequential(*blocks)


class RegnetXStage(BackboneStage):
    """
    RegNetX stage with XBlock as building block.
    """

    def build_stage(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        num_blocks: int,
        bottleneck_ratio: float,
        group_width: int,
        se_ratio: float,
        droppath_prob: float,
        **kwargs,
    ):
        group_width = self._get_divisable_group_width(out_channels, bottleneck_ratio, group_width)
        return nn.Sequential(
            XBlock(in_channels, out_channels, bottleneck_ratio, group_width, stride, se_ratio, droppath_prob),
            *[XBlock(out_channels, out_channels, bottleneck_ratio, group_width, 1, se_ratio, droppath_prob) for _ in range(num_blocks - 1)],
        )

    @staticmethod
    def _get_divisable_group_width(channels: int, bottleneck_ratio: float, group_width: int) -> int:
        """
        Returns a valid value for group_width, in channels isn't a multiplication of group_width.
        """
        inter_channels = channels // bottleneck_ratio
        # if group_width is higher than the Conv channels, fallback to a regular Conv with group_width = channels.
        if group_width > inter_channels:
            return inter_channels
        group_pow = int(math.log2(group_width))
        for pow in range(group_pow, -1, -1):
            if (inter_channels / 2**pow) % 1 == 0:
                return int(2**pow)
        return 1


class DownBlockType(Enum):
    XBlock = RegnetXStage
    REPVGG = RepVGGStage
    STDC = STDCStage


class UNetBackboneBase(AbstractUNetBackbone):
    @resolve_param("block_types_list", ListFactory(TypeFactory.from_enum_cls(DownBlockType)))
    def __init__(
        self,
        strides_list: List[int],
        width_list: List[int],
        num_blocks_list: List[int],
        block_types_list: List[Type[BackboneStage]],
        is_out_feature_list: List[bool],
        block_params: dict = {},
        in_channels: int = 3,
    ):
        super().__init__()
        self.strides_list = strides_list
        self.width_list = width_list
        self.num_blocks_list = num_blocks_list
        self.block_types_list = block_types_list
        self.is_out_feature_list = is_out_feature_list
        self.block_kwargs = block_params
        self.num_stages = len(self.strides_list)

        self.validate_backbone_arguments()

        # Build backbone stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            self.stages.append(self.block_types_list[i](in_channels, width_list[i], stride=strides_list[i], num_blocks=num_blocks_list[i], **block_params))
            in_channels = width_list[i]

    def validate_backbone_arguments(self):
        assert (
            self.num_stages == len(self.width_list) == len(self.num_blocks_list) == len(self.block_types_list) == len(self.is_out_feature_list)
        ), f"Backbone specification arguments must match to the num of stages: {self.num_stages}"

    def get_backbone_output_number_of_channels(self) -> List[int]:
        return [ch for ch, is_out in zip(self.width_list, self.is_out_feature_list) if is_out]

    def forward(self, x):
        outs = []
        for stage, is_out in zip(self.stages, self.is_out_feature_list):
            x = stage(x)
            if is_out:
                outs.append(x)
        return outs


class Encoder(nn.Module):
    def __init__(self, backbone: AbstractUNetBackbone, context_module: Optional[nn.Module]):
        super().__init__()
        self.backbone = backbone
        self.context_module = nn.Identity() if context_module is None else context_module

    def forward(self, x):
        feats = self.backbone(x)
        feats[-1] = self.context_module(feats[-1])
        return feats

    def get_output_number_of_channels(self) -> List[int]:
        """
        Return list of encoder output channels, which is backbone output channels and context module output channels in
        case the context module return different num of channels.
        """
        channels_list = self.backbone.get_backbone_output_number_of_channels()
        if hasattr(self.context_module, "out_channels") and self.context_module.out_channels is not None:
            channels_list[-1] = self.context_module.out_channels
        return channels_list


class UnetClassification(SgModule):
    def __init__(self, arch_params: HpmStruct):
        super().__init__()
        self.backbone = UNetBackboneBase(**arch_params.backbone_params)
        out_channels = self.backbone.get_backbone_output_number_of_channels()[-1]

        self.classifier_head = nn.Sequential(
            ConvBNReLU(out_channels, 1024, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(arch_params.dropout),
            nn.Linear(1024, arch_params.num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)[-1]
        return self.classifier_head(x)
