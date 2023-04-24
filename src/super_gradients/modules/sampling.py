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
