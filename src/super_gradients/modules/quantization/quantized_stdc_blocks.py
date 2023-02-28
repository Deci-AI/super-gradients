import torch
from torch import nn

from super_gradients.modules import Residual
from super_gradients.training.models.segmentation_models.stdc import (
    STDCBlock,
    AttentionRefinementModule,
    FeatureFusionModule,
    ContextPath,
)

try:
    from super_gradients.training.utils.quantization.core import SGQuantMixin, QuantizedMetadata
    from super_gradients.training.utils.quantization.selective_quantization_utils import register_quantized_module

    _imported_pytorch_quantization_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    _imported_pytorch_quantization_failure = import_err


@register_quantized_module(float_source=STDCBlock, action=QuantizedMetadata.ReplacementAction.REPLACE_AND_RECURE)
class QuantSTDCBlock(SGQuantMixin, STDCBlock):
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure

    def __init__(self, in_channels: int, out_channels: int, steps: int, stdc_downsample_mode: str, stride: int):
        super(QuantSTDCBlock, self).__init__(
            in_channels=in_channels, out_channels=out_channels, steps=steps, stdc_downsample_mode=stdc_downsample_mode, stride=stride
        )
        self.quant_list = nn.ModuleList()

        for i in range(len(self.conv_list) - 1):
            self.quant_list.append(Residual())

    def forward(self, x):
        out_list = []
        # run first conv
        x = self.conv_list[0](x)
        out_list.append(self.skip_step1(x))

        for conv, quant in zip(self.conv_list[1:], self.quant_list):
            x = conv(x)
            out_list.append(quant(x))

        out = torch.cat(out_list, dim=1)
        return out


@register_quantized_module(float_source=AttentionRefinementModule, action=QuantizedMetadata.ReplacementAction.REPLACE_AND_RECURE)
class QuantAttentionRefinementModule(SGQuantMixin, AttentionRefinementModule):
    """
    AttentionRefinementModule to apply on the last two backbone stages.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(QuantAttentionRefinementModule, self).__init__(in_channels=in_channels, out_channels=out_channels)
        self.q_x = Residual()
        self.q_y = Residual()

    def forward(self, x):
        x = self.conv_first(x)
        y = self.attention_block(x)
        return torch.mul(self.q_x(x), self.q_y(y))


@register_quantized_module(float_source=FeatureFusionModule, action=QuantizedMetadata.ReplacementAction.REPLACE_AND_RECURE)
class QuantFeatureFusionModule(SGQuantMixin, FeatureFusionModule):
    def __init__(self, spatial_channels: int, context_channels: int, out_channels: int):
        super(QuantFeatureFusionModule, self).__init__(spatial_channels=spatial_channels, context_channels=context_channels, out_channels=out_channels)

        self.q_spatial = Residual()
        self.q_context = Residual()
        self.q_feat = Residual()
        self.q_atten = Residual()

    def forward(self, spatial_feats, context_feats):
        feat = torch.cat([self.q_spatial(spatial_feats), self.q_context(context_feats)], dim=1)
        feat = self.pw_conv(feat)
        atten = self.attention_block(feat)

        feat_out = self.q_feat(feat) * self.q_atten(atten + 1)

        return feat_out


@register_quantized_module(float_source=ContextPath, action=QuantizedMetadata.ReplacementAction.REPLACE_AND_RECURE)
class QuantContextPath(SGQuantMixin, ContextPath):
    def __init__(self, backbone, fuse_channels: int, use_aux_heads: bool):
        super(QuantContextPath, self).__init__(backbone=backbone, fuse_channels=fuse_channels, use_aux_heads=use_aux_heads)
        # these extra quantizers are not necessary
        # because all the features that are inputs to them are already quantized by other modules
        # with TensorRT 8.4.1 it does no harm because they are moved and fused with other quantizers
        # they are here for other frameworks in the future

        self.q1 = Residual()
        self.q2 = Residual()
        self.q3 = Residual()
        self.q4 = Residual()

    def forward(self, x):
        feat8, feat16, feat32 = self.backbone(x)

        ce_feats = self.context_embedding(feat32)
        feat32_arm = self.arm32(feat32)
        feat32_arm = self.q1(feat32_arm) + self.q2(ce_feats)

        feat32_up = self.upsample32(feat32_arm)

        feat16_arm = self.arm16(feat16)
        feat16_arm = self.q3(feat16_arm) + self.q4(feat32_up)

        feat16_up = self.upsample16(feat16_arm)

        if self.use_aux_heads:
            return feat8, feat16_up, feat16, feat32
        return feat8, feat16_up
