from typing import Union, Optional

from torch import nn

from super_gradients.common import UpsampleMode
from super_gradients.common.data_types.enum import DownSampleMode
from super_gradients.modules import AntiAliasDownsample, PixelShuffle


def make_upsample_module(scale_factor: int, upsample_mode: Union[str, UpsampleMode], align_corners: Optional[bool] = None):
    """
    Factory method for creating upsampling modules.
    :param scale_factor: upsample scale factor
    :param upsample_mode: see UpsampleMode for supported options.
    :return: nn.Module
    """
    upsample_mode = upsample_mode.value if isinstance(upsample_mode, UpsampleMode) else upsample_mode

    if upsample_mode == UpsampleMode.NEAREST.value:
        # Prevent ValueError when passing align_corners with nearest mode.
        module = nn.Upsample(scale_factor=scale_factor, mode=upsample_mode)

    elif upsample_mode in [UpsampleMode.BILINEAR.value, UpsampleMode.BICUBIC.value]:
        module = nn.Upsample(scale_factor=scale_factor, mode=upsample_mode, align_corners=align_corners)

    elif upsample_mode == UpsampleMode.PIXEL_SHUFFLE.value:
        module = PixelShuffle(upscale_factor=scale_factor)

    elif upsample_mode == UpsampleMode.NN_PIXEL_SHUFFLE.value:
        module = nn.PixelShuffle(upscale_factor=scale_factor)
    else:
        raise NotImplementedError(f"Upsample mode: `{upsample_mode}` is not supported.")
    return module


def make_upsample_module_with_explicit_channels(
    in_channels: int, out_channels: int, scale_factor: int, upsample_mode: UpsampleMode, align_corners: Optional[bool] = None
) -> nn.Module:
    """
    Factory method for creating upsampling module with explicit control of in/out channels.
    Unlike `make_upsample_module`, this method allows to specify number of desired output channels
    which is useful for upsampling using pixel shuffle and transposed convolutions.

    :param in_channels:   Number of input channels
    :param out_channels:  Number of output channels
    :param scale_factor:  Upsample scale factor
    :param upsample_mode: The desired mode of upsampling.
    :param align_corners: See `nn.Upsample` for details.
    :return:              Created upsampling module.
    """
    projection_before_upsample = None

    if upsample_mode == UpsampleMode.NEAREST:
        # Prevent ValueError when passing align_corners with nearest mode.
        module = nn.Upsample(scale_factor=scale_factor, mode=upsample_mode)
        if in_channels != out_channels:
            projection_before_upsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    elif upsample_mode in [UpsampleMode.BILINEAR.value, UpsampleMode.BICUBIC.value]:
        module = nn.Upsample(scale_factor=scale_factor, mode=upsample_mode, align_corners=align_corners)

    elif upsample_mode == UpsampleMode.PIXEL_SHUFFLE:
        if in_channels != out_channels * scale_factor**2:
            projection_before_upsample = nn.Conv2d(in_channels, out_channels * scale_factor**2, kernel_size=1, stride=1, padding=0, bias=False)
        module = PixelShuffle(upscale_factor=scale_factor)

    elif upsample_mode == UpsampleMode.NN_PIXEL_SHUFFLE:
        if in_channels != out_channels * scale_factor**2:
            projection_before_upsample = nn.Conv2d(in_channels, out_channels * scale_factor**2, kernel_size=1, stride=1, padding=0, bias=False)
        module = nn.PixelShuffle(upscale_factor=scale_factor)

    elif upsample_mode == UpsampleMode.CONV_TRANSPOSE:
        module = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor)
    else:
        raise NotImplementedError(f"Upsample mode: `{upsample_mode}` is not supported.")

    if projection_before_upsample is not None:
        module = nn.Sequential(projection_before_upsample, module)

    return module


def make_downsample_module(in_channels: int, stride: int, downsample_mode: Union[str, DownSampleMode]):
    """
    Factory method for creating down-sampling modules.
    :param downsample_mode: see DownSampleMode for supported options.
    :return: nn.Module
    """
    downsample_mode = downsample_mode.value if isinstance(downsample_mode, DownSampleMode) else downsample_mode
    if downsample_mode == DownSampleMode.ANTI_ALIAS.value:
        return AntiAliasDownsample(in_channels, stride)
    if downsample_mode == DownSampleMode.MAX_POOL.value:
        return nn.MaxPool2d(kernel_size=stride, stride=stride)
    raise NotImplementedError(f"DownSample mode: `{downsample_mode}` is not supported.")
