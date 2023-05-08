import math
from typing import List, Type, Optional, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from super_gradients.common.registry.registry import register_model, register_unet_backbone_stage, BACKBONE_STAGES
from super_gradients.common.object_names import Models
from super_gradients.common.factories.context_modules_factory import ContextModulesFactory
from super_gradients.training.models.segmentation_models.context_modules import AbstractContextModule
from super_gradients.training.utils.utils import get_param, HpmStruct
from super_gradients.training import models
from super_gradients.training.models.classification_models.regnet import XBlock
from super_gradients.training.models.classification_models.repvgg import RepVGGBlock
from super_gradients.training.models.segmentation_models.stdc import STDCBlock
from super_gradients.training.models import SgModule
from super_gradients.modules import ConvBNReLU, QARepVGGBlock
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.factories.type_factory import TypeFactory
from super_gradients.common.data_types.enum import DownSampleMode
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.modules.sampling import make_downsample_module

logger = get_logger(__name__)


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

    @abstractmethod
    def get_all_number_of_channels(self) -> List[int]:
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


@register_unet_backbone_stage()
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


class ConvBaseStage(BackboneStage, ABC):
    """
    Base single conv block implementation, such as, Conv, QARepVGG, and RepVGG stages.
    Optionally support different downsample strategy, `anti_alias` with the `AntiAliasDownsample` and `max_pool` with
    the `nn.MaxPool2d` module.
    """

    def build_stage(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        num_blocks: int,
        anti_alias: bool,
        downsample_mode: Optional[Union[str, DownSampleMode]] = None,
        **kwargs,
    ):
        blocks = []
        # Init down-sample module
        if anti_alias:
            logger.warning("`anti_alias` argument is deprecated and will be removed in future versions.")
            if downsample_mode is not None:
                raise ValueError(f"Only one argument should set as downsample_mode found: anti_alias: `True`," f" and downsample_mode: {downsample_mode}.")
            downsample_mode = DownSampleMode.ANTI_ALIAS

        if downsample_mode is not None and stride == 2:
            blocks.append(make_downsample_module(in_channels, stride=stride, downsample_mode=downsample_mode))
            stride = 1

        # RepVGG blocks
        blocks.extend(
            [
                self.build_conv_block(in_channels, out_channels, stride=stride),
                *[self.build_conv_block(out_channels, out_channels, stride=1) for _ in range(num_blocks - 1)],
            ]
        )
        return nn.Sequential(*blocks)

    @abstractmethod
    def build_conv_block(self, in_channels: int, out_channels: int, stride: int):
        raise NotImplementedError()


@register_unet_backbone_stage()
class RepVGGStage(ConvBaseStage):
    """
    RepVGG stage with RepVGGBlock as building block.
    """

    def build_conv_block(self, in_channels: int, out_channels: int, stride: int):
        return RepVGGBlock(in_channels, out_channels, stride=stride)


@register_unet_backbone_stage()
class QARepVGGStage(ConvBaseStage):
    """
    QARepVGG stage with QARepVGGBlock as building block.
    """

    def build_conv_block(self, in_channels: int, out_channels: int, stride: int):
        return QARepVGGBlock(in_channels, out_channels, stride=stride, use_residual_connection=(out_channels == in_channels and stride == 1))


@register_unet_backbone_stage()
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


@register_unet_backbone_stage()
class ConvStage(ConvBaseStage):
    """
    Conv stage with ConvBNReLU as building block.
    """

    def build_conv_block(self, in_channels: int, out_channels: int, stride: int):
        return ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class UNetBackboneBase(AbstractUNetBackbone):
    @resolve_param("block_types_list", ListFactory(TypeFactory(BACKBONE_STAGES)))
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

    def get_all_number_of_channels(self) -> List[int]:
        return self.width_list

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

    def get_all_number_of_channels(self) -> List[int]:
        channels_list = self.backbone.get_all_number_of_channels()
        if hasattr(self.context_module, "output_channels"):
            channels_list[-1] = self.context_module.output_channels()
        return channels_list


class UnetClassification(SgModule):
    @resolve_param("context_module", ContextModulesFactory())
    def __init__(
        self,
        num_classes: int,
        backbone_params: dict,
        context_module: AbstractContextModule,
        dropout: float,
    ):
        super().__init__()
        backbone = UNetBackboneBase(**backbone_params)

        self.encoder = Encoder(backbone, context_module)
        out_channels = self.encoder.get_output_number_of_channels()[-1]

        self.classifier_head = nn.Sequential(
            ConvBNReLU(out_channels, 1024, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.encoder(x)[-1]
        return self.classifier_head(x)


@register_model(Models.UNET_CUSTOM_CLS)
class UnetClassificationCustom(UnetClassification):
    def __init__(self, arch_params: HpmStruct):
        arch_params = HpmStruct(**models.get_arch_params("unet_default_arch_params.yaml", overriding_params=arch_params.to_dict()))
        super().__init__(
            num_classes=get_param(arch_params, "num_classes"),
            backbone_params=get_param(arch_params, "backbone_params"),
            context_module=get_param(arch_params, "context_module", nn.Identity()),
            dropout=get_param(arch_params, "dropout", 0.0),
        )
