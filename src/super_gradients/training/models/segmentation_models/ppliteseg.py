import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, List, Tuple
from super_gradients.training.utils.module_utils import ConvBNReLU, UpsampleMode, make_upsample_module
from super_gradients.training.models.segmentation_models.stdc import SegmentationHead, AbstractSTDCBackbone, STDC1Backbone, STDC2Backbone
from super_gradients.training.models.segmentation_models.segmentation_module import SegmentationModule
from super_gradients.training.utils import HpmStruct, get_param


class PPLiteSegEncoder(nn.Module):
    def __init__(self,
                 backbone: AbstractSTDCBackbone,
                 projection_channels_list: List[int],
                 context_module: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.context_module = context_module
        feats_channels = backbone.get_backbone_output_number_of_channels()
        self.proj_convs = nn.ModuleList([
            ConvBNReLU(feat_ch, proj_ch, kernel_size=3, padding=1, bias=False)
            for feat_ch, proj_ch in zip(feats_channels, projection_channels_list)
        ])

    def get_output_number_of_channels(self) -> List[int]:
        channels_list = self.backbone.get_backbone_output_number_of_channels()
        if hasattr(self.context_module, "output_channels"):
            channels_list[-1] = self.context_module.output_channels()
        return channels_list

    def forward(self, x):
        feats = self.backbone(x)
        y = self.context_module(feats[-1])
        feats = [conv(f) for conv, f in zip(self.proj_convs, feats)]
        return feats + [y]


class PPLiteSegDecoder(nn.Module):
    def __init__(self,
                 encoder_channels: List[int],
                 up_factors: List[int],
                 out_channels: List[int],
                 upsample_mode,
                 align_corners: bool):
        super().__init__()
        encoder_channels.reverse()
        in_channels = encoder_channels.pop(0)
        # TODO - assert argument length
        self.up_stages = nn.ModuleList()
        for skip_ch, up_factor, out_ch in zip(encoder_channels, up_factors, out_channels):
            self.up_stages.append(UAFM(
                in_channels=in_channels,
                skip_channels=skip_ch,
                out_channels=out_ch,
                up_factor=up_factor,
                upsample_mode=upsample_mode,
                align_corners=align_corners
            ))
            in_channels = out_ch

    def forward(self, feats: List[torch.Tensor]):
        feats.reverse()
        x = feats.pop(0)
        for up_stage, skip in zip(self.up_stages, feats):
            x = up_stage(x, skip)
        return x


class PPLiteSegBase(SegmentationModule):
    """
    The PP_LiteSeg implementation based on PaddlePaddle.
    The original article refers to "Juncai Peng, Yi Liu, Shiyu Tang, Yuying Hao, Lutao Chu,
    Guowei Chen, Zewu Wu, Zeyu Chen, Zhiliang Yu, Yuning Du, Qingqing Dang,Baohua Lai,
    Qiwen Liu, Xiaoguang Hu, Dianhai Yu, Yanjun Ma. PP-LiteSeg: A Superior Real-Time Semantic
    Segmentation Model. https://arxiv.org/abs/2204.02681".
    """
    def __init__(self,
                 num_classes,
                 backbone: AbstractSTDCBackbone,
                 projection_channels_list: List[int],
                 sppm_inter_channels: int,
                 sppm_out_channels: int,
                 sppm_pool_sizes: List[int],
                 sppm_upsample_mode: Union[UpsampleMode, str],
                 align_corners: bool,
                 decoder_up_factors: List[int],
                 decoder_channels: List[int],
                 decoder_upsample_mode: Union[UpsampleMode, str],
                 head_upsample_mode: Union[UpsampleMode, str],
                 head_mid_channels: int,
                 dropout: float,
                 use_aux_heads: bool,
                 aux_hidden_channels: List[int],
                 aux_scale_factors: List[int]
                 ):
        super().__init__(use_aux_heads=use_aux_heads)

        # backbone
        # assert hasattr(backbone, 'feat_channels'), \
        #     "The backbone should has feat_channels."
        # assert len(backbone.feat_channels) >= len(backbone_indices), \
        #     f"The length of input backbone_indices ({len(backbone_indices)}) should not be" \
        #     f"greater than the length of feat_channels ({len(backbone.feat_channels)})."
        # assert len(backbone.feat_channels) > max(backbone_indices), \
        #     f"The max value ({max(backbone_indices)}) of backbone_indices should be " \
        #     f"less than the length of feat_channels ({len(backbone.feat_channels)})."
        context = SPPM(in_channels=backbone.get_backbone_output_number_of_channels()[-1],
                       inter_channels=sppm_inter_channels,
                       out_channels=sppm_out_channels,
                       pool_sizes=sppm_pool_sizes,
                       upsample_mode=sppm_upsample_mode,
                       align_corners=align_corners)
        self.encoder = PPLiteSegEncoder(backbone=backbone,
                                        context_module=context,
                                        projection_channels_list=projection_channels_list)
        encoder_channels = projection_channels_list + [sppm_out_channels]
        self.decoder = PPLiteSegDecoder(encoder_channels=encoder_channels,
                                        up_factors=decoder_up_factors,
                                        out_channels=decoder_channels,
                                        upsample_mode=decoder_upsample_mode,
                                        align_corners=align_corners)
        self.seg_head = nn.Sequential(
            SegmentationHead(in_channels=decoder_channels[-1],
                             mid_channels=head_mid_channels,
                             num_classes=num_classes,
                             dropout=dropout),
            make_upsample_module(scale_factor=8, upsample_mode=head_upsample_mode, align_corners=align_corners)
        )
        # Auxiliary heads
        if self.use_aux_heads:
            encoder_out_channels = projection_channels_list
            self.aux_heads = nn.ModuleList([
                nn.Sequential(
                    SegmentationHead(backbone_ch, hidden_ch, num_classes, dropout=dropout),
                    make_upsample_module(scale_factor=scale_factor, upsample_mode=head_upsample_mode,
                                         align_corners=align_corners)
                ) for backbone_ch, hidden_ch, scale_factor in zip(encoder_out_channels, aux_hidden_channels,
                                                                  aux_scale_factors)
            ])
        self.init_params()

    def _remove_auxiliary_heads(self):
        if hasattr(self, "aux_heads"):
            del self.aux_heads

    @property
    def backbone(self) -> nn.Module:
        return self.encoder.backbone

    def forward(self, x):
        feats = self.encoder(x)
        if self.use_aux_heads:
            enc_feats = feats[:-1]
        x = self.decoder(feats)
        x = self.seg_head(x)
        if not self.use_aux_heads:
            return x
        aux_feats = [aux_head(feat) for feat, aux_head in zip(enc_feats, self.aux_heads)]
        return tuple([x] + aux_feats)


class SPPM(nn.Module):
    """
    Simple Pyramid Pooling context Module.
    """
    def __init__(self,
                 in_channels: int,
                 inter_channels: int,
                 out_channels: int,
                 pool_sizes: List[Union[int, Tuple[int, int]]],
                 upsample_mode: Union[UpsampleMode, str] = UpsampleMode.BILINEAR,
                 align_corners: bool = False):
        """
        :param inter_channels: num channels in each pooling branch.
        :param out_channels: The number of output channels after pyramid pooling module.
        :param pool_sizes: output sizes of the pooled feature maps.
        """
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                ConvBNReLU(in_channels, inter_channels, kernel_size=1, bias=False),
            ) for pool_size in pool_sizes
        ])
        self.conv_out = ConvBNReLU(inter_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

    def output_channels(self):
        return self.out_channels

    def forward(self, x):
        out = None
        input_shape = x.shape[2:]
        for branch in self.branches:
            y = branch(x)
            y = F.interpolate(y, size=input_shape, mode=self.upsample_mode, align_corners=self.align_corners)
            out = y if out is None else out + y
        out = self.conv_out(out)
        return out


class UAFM(nn.Module):
    """
    Unified Attention Fusion Module, which uses mean and max values across the spatial dimensions.
    """
    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 up_factor: int,
                 upsample_mode: Union[UpsampleMode, str] = UpsampleMode.BILINEAR,
                 align_corners: bool = False):
        """
        TODO - doc
        """
        super().__init__()
        self.conv_atten = nn.Sequential(
            ConvBNReLU(4, 2, kernel_size=3, padding=1, bias=False),
            ConvBNReLU(2, 1, kernel_size=3, padding=1, bias=False, use_activation=False)
        )

        self.proj_skip = nn.Identity() if skip_channels == in_channels else \
            ConvBNReLU(skip_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.up_x = nn.Identity() if up_factor == 1 else \
            make_upsample_module(scale_factor=up_factor, upsample_mode=upsample_mode, align_corners=align_corners)
        self.conv_out = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x, skip):
        """
        TODO - doc
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        x = self.up_x(x)
        skip = self.proj_skip(skip)

        atten = torch.cat([
            *self._avg_max_spatial_reduce(x, use_concat=False),
            *self._avg_max_spatial_reduce(skip, use_concat=False)
        ], dim=1)
        atten = self.conv_atten(atten)
        atten = torch.sigmoid(atten)

        out = x * atten + skip * (1 - atten)
        out = self.conv_out(out)
        return out

    @staticmethod
    def _avg_max_spatial_reduce(x, use_concat: bool = False):
        reduced = [
            torch.mean(x, dim=1, keepdim=True),
            torch.max(x, dim=1, keepdim=True)[0]
        ]
        if use_concat:
            reduced = torch.cat(reduced, dim=1)
        return reduced


class PPLiteSegB(PPLiteSegBase):
    def __init__(self, arch_params: HpmStruct):
        backbone = STDC2Backbone(in_channels=3, out_down_ratios=[8, 16, 32])
        super().__init__(num_classes=get_param(arch_params, "num_classes"),
                         backbone=backbone,
                         projection_channels_list=[96, 128, 128],
                         sppm_inter_channels=128,
                         sppm_out_channels=128,
                         sppm_pool_sizes=[1, 2, 4],
                         sppm_upsample_mode="bilinear",
                         align_corners=False,
                         decoder_up_factors=[1, 2, 2],
                         decoder_channels=[128, 96, 64],
                         decoder_upsample_mode="bilinear",
                         head_upsample_mode="bilinear",
                         head_mid_channels=64,
                         dropout=get_param(arch_params, "dropout", 0.),
                         use_aux_heads=get_param(arch_params, "use_aux_heads", False),
                         aux_hidden_channels=[32, 64, 64],
                         aux_scale_factors=[8, 16, 32])


class PPLiteSegT(PPLiteSegBase):
    def __init__(self, arch_params: HpmStruct):
        backbone = STDC1Backbone(in_channels=3, out_down_ratios=[8, 16, 32])
        super().__init__(num_classes=get_param(arch_params, "num_classes"),
                         backbone=backbone,
                         projection_channels_list=[64, 128, 128],
                         sppm_inter_channels=128,
                         sppm_out_channels=128,
                         sppm_pool_sizes=[1, 2, 4],
                         sppm_upsample_mode="bilinear",
                         align_corners=False,
                         decoder_up_factors=[1, 2, 2],
                         decoder_channels=[128, 64, 32],
                         decoder_upsample_mode="bilinear",
                         head_upsample_mode="bilinear",
                         head_mid_channels=32,
                         dropout=get_param(arch_params, "dropout", 0.),
                         use_aux_heads=get_param(arch_params, "use_aux_heads", False),
                         aux_hidden_channels=[32, 64, 64],
                         aux_scale_factors=[8, 16, 32])


if __name__ == '__main__':
    m = PPLiteSegT(HpmStruct(num_classes=19, use_aux_heads=True))
    x = torch.randn(2, 3, 1024, 2048)

    def print_outputs(y):
        if isinstance(y, torch.Tensor):
            print(y.shape)
        else:
            for ys in y:
                print(ys.shape)

    print_outputs(m(x))

    torch.onnx.export(m, x, "/Users/liork/Downloads/tmp.onnx", opset_version=11)
