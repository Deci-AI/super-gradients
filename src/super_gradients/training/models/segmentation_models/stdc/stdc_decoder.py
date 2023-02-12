from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from super_gradients.modules import ConvBNReLU


class AttentionRefinementModule(nn.Module):
    """
    AttentionRefinementModule to apply on the last two backbone stages.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(AttentionRefinementModule, self).__init__()
        self.conv_first = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.attention_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), ConvBNReLU(out_channels, out_channels, kernel_size=1, bias=False, use_activation=False), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_first(x)
        y = self.attention_block(x)
        return torch.mul(x, y)


class FeatureFusionModule(nn.Module):
    """
    Fuse features from higher resolution aka, spatial feature map with features from lower resolution with high
     semantic information aka, context feature map.
    :param spatial_channels: num channels of input from spatial path.
    :param context_channels: num channels of input from context path.
    :param out_channels: num channels of feature fusion module.
    :param projection_channels: num channels to project both spacial and context feats before concatenation using
        1x1 convolution. Projection convolutions are applied only if spatial / context channels are above the
        projection channels. It has been proved to reduce the memory bottleneck and reduce latency with minor drop
        in accuracy. Default is `None` to not apply projection convolutions.
    """

    def __init__(self, spatial_channels: int, context_channels: int, out_channels: int, projection_channels: int = None):
        super(FeatureFusionModule, self).__init__()

        self.proj_spatial, self.proj_context = nn.Identity(), nn.Identity()
        if projection_channels is not None and spatial_channels > projection_channels:
            self.proj_spatial = ConvBNReLU(spatial_channels, projection_channels, kernel_size=1, stride=1, bias=False)
            spatial_channels = projection_channels

        if projection_channels is not None and context_channels > projection_channels:
            self.proj_context = ConvBNReLU(context_channels, projection_channels, kernel_size=1, stride=1, bias=False)
            context_channels = projection_channels

        self.pw_conv = ConvBNReLU(spatial_channels + context_channels, out_channels, kernel_size=1, stride=1, bias=False)
        # TODO - used without bias in convolutions by mistake, try to reproduce with bias=True
        self.attention_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_channels=out_channels, out_channels=out_channels // 4, kernel_size=1, use_normalization=False, bias=False),
            nn.Conv2d(in_channels=out_channels // 4, out_channels=out_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, spatial_feats, context_feats):
        spatial_feats = self.proj_spatial(spatial_feats)
        context_feats = self.proj_context(context_feats)

        feat = torch.cat([spatial_feats, context_feats], dim=1)
        feat = self.pw_conv(feat)
        atten = self.attention_block(feat)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class ContextEmbeddingOnline(nn.Module):
    """
    ContextEmbedding module that use global average pooling to 1x1 to extract context information, and then upsample
    to original input size.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ContextEmbeddingOnline, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_embedding = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        out_height, out_width = x.size()[2:]
        x = self.context_embedding(x)
        return F.interpolate(x, size=(out_height, out_width), mode="nearest")


class ContextEmbeddingFixedSize(ContextEmbeddingOnline):
    """
    ContextEmbedding module that use a fixed size interpolation, supported with onnx conversion.
    Prevent slice/cast/shape operations in onnx conversion for applying interpolation.
    """

    def __init__(self, in_channels: int, out_channels: int, upsample_size: Union[list, tuple]):
        super(ContextEmbeddingFixedSize, self).__init__(in_channels, out_channels)
        self.context_embedding.add_module("upsample", nn.Upsample(scale_factor=upsample_size, mode="nearest"))

    @classmethod
    def from_context_embedding_online(cls, ce_online: ContextEmbeddingOnline, upsample_size: Union[list, tuple]):
        context = ContextEmbeddingFixedSize(in_channels=ce_online.in_channels, out_channels=ce_online.out_channels, upsample_size=upsample_size)
        # keep training mode state as original module
        context.train(ce_online.training)
        context.load_state_dict(ce_online.state_dict())
        return context

    def forward(self, x):
        return self.context_embedding(x)


class STDCDecoder(nn.Module):
    def __init__(self, skip_channels_list: list, context_fuse_channels: int, ffm_channels: int, ffm_projection_channels: Optional[int]):
        super().__init__()
        assert len(skip_channels_list) == 3, f"{self.__class__.__name__} support only 3 outputs from the encoder"
        # get num of channels for two last stages
        channels8, channels16, channels32 = skip_channels_list
        self.context_embedding = ContextEmbeddingOnline(channels32, context_fuse_channels)
        self.arm32 = AttentionRefinementModule(channels32, context_fuse_channels)
        self.upsample32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBNReLU(context_fuse_channels, context_fuse_channels, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.arm16 = AttentionRefinementModule(channels16, context_fuse_channels)
        self.upsample16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBNReLU(context_fuse_channels, context_fuse_channels, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.ffm = FeatureFusionModule(
            spatial_channels=channels8, context_channels=context_fuse_channels, out_channels=ffm_channels, projection_channels=ffm_projection_channels
        )
        self.out_channels = ffm_channels

    def forward(self, feats):
        feat8, feat16, feat32 = feats

        ce_feats = self.context_embedding(feat32)
        feat32_arm = self.arm32(feat32)
        feat32_arm = feat32_arm + ce_feats

        feat32_up = self.upsample32(feat32_arm)

        feat16_arm = self.arm16(feat16)
        feat16_arm = feat16_arm + feat32_up
        feat16_up = self.upsample16(feat16_arm)

        feat_out = self.ffm(spatial_feats=feat8, context_feats=feat16_up)
        return feat_out

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        """
        Replace ContextEmbeddingOnline which cause compilation issues and not supported in some compilations, to ContextEmbeddingFixedSize.
        """

        context_embedding_up_size = (input_size[-2] // 32, input_size[-1] // 32)
        self.context_embedding = ContextEmbeddingFixedSize.from_context_embedding_online(self.context_embedding, context_embedding_up_size)
