from typing import List, Type
from abc import ABC, abstractmethod
from enum import Enum

import torch.nn as nn
import torch

from super_gradients.modules import ConvBNReLU
from super_gradients.training.utils.module_utils import make_upsample_module
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.factories.type_factory import TypeFactory


class AbstractUpFuseBlock(nn.Module, ABC):
    """
    Abstract class for upsample and fuse UNet decoder building block.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, **kwargs):
        """
        :param in_channels: num_channels of the feature map to be upsample.
        :param skip_channels: num_channels of the skip feature map from higher resolution.
        :param out_channels: num_channels of the output features.
        """
        super().__init__()

    @abstractmethod
    def forward(self, x, skip):
        raise NotImplementedError()


class UpFactorBlock(AbstractUpFuseBlock):
    """
    Ignore Skip features, simply apply upsampling and ConvBNRelu layers.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, up_factor: int, mode: str, num_repeats: int, **kwargs):
        super().__init__(in_channels=in_channels, skip_channels=0, out_channels=out_channels)
        self.up_path = make_upsample_module(scale_factor=up_factor, upsample_mode=mode, align_corners=False)

        self.last_convs = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Sequential(*[ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1, bias=False) for _ in range(num_repeats - 1)]),
        )

    def forward(self, x, skip):
        x = self.up_path(x)
        return self.last_convs(x)


class UpCatBlock(AbstractUpFuseBlock):
    """
    Fuse features with concatenation and followed Convolutions.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, up_factor: int, mode: str, num_repeats: int, **kwargs):
        super().__init__(in_channels=in_channels, skip_channels=skip_channels, out_channels=out_channels)
        self.up_path = make_upsample_module(scale_factor=up_factor, upsample_mode=mode, align_corners=False)
        self.last_convs = nn.Sequential(
            ConvBNReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Sequential(*[ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1, bias=False) for _ in range(num_repeats - 1)]),
        )

    def forward(self, x, skip):
        x = self.up_path(x)
        x = torch.cat([x, skip], dim=1)
        return self.last_convs(x)


class UpBlockType(Enum):
    UP_FACTOR = UpFactorBlock
    UP_CAT = UpCatBlock


class Decoder(nn.Module):
    @resolve_param("up_block_types", ListFactory(TypeFactory.from_enum_cls(UpBlockType)))
    def __init__(
        self,
        skip_channels_list: List[int],
        up_block_repeat_list: List[int],
        skip_expansion: float,
        decoder_scale: float,
        up_block_types: List[Type[AbstractUpFuseBlock]],
        is_skip_list: List[bool],
        min_decoder_channels: int = 1,
        **up_block_kwargs,
    ):
        """

        :param skip_channels_list: num_channels list of skip feature maps from the encoder.
        :param up_block_repeat_list: `num_repeats` arg list to be passed to the UpFuseBlocks.
        :param skip_expansion: skip expansion ratio value, before fusing the skip features from the encoder with the
            decoder features, a projection convolution is applied upon the encoder features to project the num_channels
            by skip_expansion as follows: `num_channels = skip_channels * skip_expansion
        :param decoder_scale: num_channels width ratio between encoder stages and decoder stages.
        :param min_decoder_channels: The minimum num_channels of decoder stages. Useful i.e if we want to keep the width
            above the num of classes. The num_channels of a decoder stage is determined as follows:
                `decoder_channels = max(encoder_channels * decoder_scale, min_decoder_channels)`
        :param up_block_types: list of AbstractUpFuseBlock.
        :param is_skip_list: List of flags whether to use feature-map from encoder stage as skip connection or not. Used
            to not apply projection convolutions if a certain encoder feature is not aggregate with the decoder.
        :param up_block_kwargs: init parameters for fuse blocks.
        """
        super().__init__()
        # num_channels list after encoder features projections.
        self.up_channels_list = [max(int(ch * decoder_scale), min_decoder_channels) for ch in skip_channels_list]
        # Reverse order to up-bottom order, i.e [stage4_ch, stage3_ch, ... , stage1_ch]
        self.up_channels_list.reverse()
        # Remove last stage num_channels, as it is the input to the decoder.
        self.up_channels_list.pop(0)

        is_skip_list.reverse()
        is_skip_list += [False]

        self.projection_blocks, skip_channels_list = self._make_skip_projection(skip_channels_list, skip_expansion, is_skip_list, min_decoder_channels)
        skip_channels_list = skip_channels_list.copy()
        skip_channels_list.reverse()

        self.up_stages = nn.ModuleList()
        in_channels = skip_channels_list.pop(0)
        skip_channels_list.append(None)
        for i in range(len(up_block_types)):
            self.up_stages.append(
                up_block_types[i](in_channels, skip_channels_list[i], self.up_channels_list[i], num_repeats=up_block_repeat_list[i], **up_block_kwargs)
            )
            in_channels = self.up_channels_list[i]

    def _make_skip_projection(self, skip_channels_list: list, skip_expansion: float, is_skip_list: list, min_decoder_channels: int):
        if skip_expansion == 1.0:
            return nn.ModuleList([nn.Identity()] * len(skip_channels_list)), skip_channels_list

        projection_channels = [max(int(ch * skip_expansion), min_decoder_channels) for ch in skip_channels_list]
        blocks = nn.ModuleList()
        for i in range(len(skip_channels_list)):
            if not is_skip_list[i]:
                blocks.append(nn.Identity())
                projection_channels[i] = skip_channels_list[i]
            else:
                blocks.append(ConvBNReLU(skip_channels_list[i], projection_channels[i], kernel_size=1, bias=False, use_activation=False))

        return blocks, projection_channels

    def forward(self, feats: List[torch.Tensor]):
        feats = [adapt_conv(feat) for feat, adapt_conv in zip(feats, self.projection_blocks)]
        # Reverse order to up-bottom order, i.e [stage4_ch, stage3_ch, ... , stage1_ch]
        feats.reverse()
        # Remove last stage feature map, as it is the input to the decoder and not a skip connection.
        x = feats.pop(0)
        for up_stage, skip in zip(self.up_stages, feats):
            x = up_stage(x, skip)
        return x
