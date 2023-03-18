from typing import List, Type, Union, Optional, Tuple
from abc import ABC, abstractmethod

import torch.nn as nn
import torch

from super_gradients.common.registry.registry import register_unet_up_block, UP_FUSE_BLOCKS
from super_gradients.modules import ConvBNReLU, CrossModelSkipConnection, Residual
from super_gradients.modules.sampling import make_upsample_module
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.factories.type_factory import TypeFactory
from super_gradients.common import UpsampleMode


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

    @staticmethod
    def validate_upsample_mode(
        in_channels: int, up_factor: int, upsample_mode: Union[UpsampleMode, str], fallback_mode: Optional[Union[UpsampleMode, str]] = None
    ) -> Tuple[Union[UpsampleMode, str], int]:
        """
        Validate whether the upsample_mode is supported, and returns the upsample path output channels.
        :return: tuple of upsample_mode and out_channels of the upsample module
        """
        out_channels = in_channels
        upsample_mode = upsample_mode.value if isinstance(upsample_mode, UpsampleMode) else upsample_mode
        if upsample_mode in [UpsampleMode.PIXEL_SHUFFLE.value, UpsampleMode.NN_PIXEL_SHUFFLE.value]:
            # Check if in_channels is divisible by (up_factor ** 2) for pixel shuffle, else fallback to fallback_mode.
            _in_ch = in_channels / (up_factor**2)
            if _in_ch % 1 == 0:
                out_channels = int(_in_ch)
            elif fallback_mode is not None:
                upsample_mode = fallback_mode
            else:
                raise ValueError(
                    f"Upsample mode: {upsample_mode} can't be used, due to in_channels: {in_channels} "
                    f"is not divisible by (up_factor ** 2) for up_factor: {up_factor}.\n"
                    f"Consider setting a `fallback_mode`."
                )
        return upsample_mode, out_channels


@register_unet_up_block()
class UpFactorBlock(AbstractUpFuseBlock):
    """
    Ignore Skip features, simply apply upsampling and ConvBNRelu layers.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        up_factor: int,
        mode: Union[UpsampleMode, str],
        num_repeats: int,
        fallback_mode: Optional[Union[UpsampleMode, str]] = None,
        **kwargs,
    ):
        super().__init__(in_channels=in_channels, skip_channels=0, out_channels=out_channels)

        mode, up_out_channels = self.validate_upsample_mode(in_channels, up_factor=up_factor, upsample_mode=mode, fallback_mode=fallback_mode)
        self.up_path = make_upsample_module(scale_factor=up_factor, upsample_mode=mode, align_corners=False)

        self.last_convs = nn.Sequential(
            ConvBNReLU(up_out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Sequential(*[ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1, bias=False) for _ in range(num_repeats - 1)]),
        )

    def forward(self, x, skip):
        x = self.up_path(x)
        return self.last_convs(x)


@register_unet_up_block()
class UpCatBlock(AbstractUpFuseBlock):
    """
    Fuse features with concatenation and followed Convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        up_factor: int,
        mode: Union[UpsampleMode, str],
        num_repeats: int,
        fallback_mode: Optional[Union[UpsampleMode, str]] = None,
        **kwargs,
    ):
        super().__init__(in_channels=in_channels, skip_channels=skip_channels, out_channels=out_channels)

        mode, up_out_channels = self.validate_upsample_mode(in_channels, up_factor=up_factor, upsample_mode=mode, fallback_mode=fallback_mode)

        self.up_path = make_upsample_module(scale_factor=up_factor, upsample_mode=mode, align_corners=False)

        self.last_convs = nn.Sequential(
            ConvBNReLU(up_out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Sequential(*[ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1, bias=False) for _ in range(num_repeats - 1)]),
        )

    def forward(self, x, skip):
        x = self.up_path(x)
        x = torch.cat([x, skip], dim=1)
        return self.last_convs(x)


@register_unet_up_block()
class UpSumBlock(AbstractUpFuseBlock):
    """
    Fuse features with concatenation and followed Convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        up_factor: int,
        mode: Union[UpsampleMode, str],
        num_repeats: int,
        fallback_mode: Optional[Union[UpsampleMode, str]] = None,
        **kwargs,
    ):
        super().__init__(in_channels=in_channels, skip_channels=skip_channels, out_channels=out_channels)
        mode, up_out_channels = self.validate_upsample_mode(in_channels, up_factor=up_factor, upsample_mode=mode, fallback_mode=fallback_mode)

        self.up_path = make_upsample_module(scale_factor=up_factor, upsample_mode=mode, align_corners=False)

        self.proj_conv = (
            Residual() if skip_channels == up_out_channels else ConvBNReLU(skip_channels, up_out_channels, kernel_size=1, bias=False, use_activation=False)
        )

        self.last_convs = nn.Sequential(
            ConvBNReLU(up_out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Sequential(*[ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1, bias=False) for _ in range(num_repeats - 1)]),
        )

    def forward(self, x, skip):
        skip = self.proj_conv(skip)
        x = self.up_path(x)
        x = x + skip
        return self.last_convs(x)


class Decoder(nn.Module):
    @resolve_param("up_block_types", ListFactory(TypeFactory(UP_FUSE_BLOCKS)))
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
            return nn.ModuleList([CrossModelSkipConnection()] * len(skip_channels_list)), skip_channels_list

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
