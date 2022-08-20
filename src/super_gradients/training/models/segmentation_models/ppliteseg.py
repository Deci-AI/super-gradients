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


class PPLiteSeg(nn.Module):
    """
    The PP_LiteSeg implementation based on PaddlePaddle.

    The original article refers to "Juncai Peng, Yi Liu, Shiyu Tang, Yuying Hao, Lutao Chu,
    Guowei Chen, Zewu Wu, Zeyu Chen, Zhiliang Yu, Yuning Du, Qingqing Dang,Baohua Lai,
    Qiwen Liu, Xiaoguang Hu, Dianhai Yu, Yanjun Ma. PP-LiteSeg: A Superior Real-Time Semantic
    Segmentation Model. https://arxiv.org/abs/2204.02681".

    Args:
        num_classes (int): The number of target classes.
        backbone(nn.Layer): Backbone network, such as stdc1net and resnet18. The backbone must
            has feat_channels, of which the length is 5.
        backbone_indices (List(int), optional): The values indicate the indices of output of backbone.
            Default: [2, 3, 4].
        arm_type (str, optional): The type of attention refinement module. Default: ARM_Add_SpAttenAdd3.
        cm_bin_sizes (List(int), optional): The bin size of context module. Default: [1,2,4].
        cm_out_ch (int, optional): The output channel of the last context module. Default: 128.
        arm_out_chs (List(int), optional): The out channels of each arm module. Default: [64, 96, 128].
        seg_head_inter_chs (List(int), optional): The intermediate channels of segmentation head.
            Default: [64, 64, 64].
        resize_mode (str, optional): The resize mode for the upsampling operation in decoder.
            Default: bilinear.
        pretrained (str, optional): The path or url of pretrained model. Default: None.

    """

    def __init__(self,
                 num_classes,
                 backbone: AbstractSTDCBackbone,
                 backbone_indices=[2, 3, 4],
                 arm_type='UAFM_SpAtten',
                 cm_bin_sizes=[1, 2, 4],
                 cm_out_ch=128,
                 arm_out_chs=[64, 96, 128],
                 seg_head_inter_chs=[64, 64, 64],
                 resize_mode='bilinear',
                 pretrained=None):
        super().__init__()

        # backbone
        assert hasattr(backbone, 'feat_channels'), \
            "The backbone should has feat_channels."
        assert len(backbone.feat_channels) >= len(backbone_indices), \
            f"The length of input backbone_indices ({len(backbone_indices)}) should not be" \
            f"greater than the length of feat_channels ({len(backbone.feat_channels)})."
        assert len(backbone.feat_channels) > max(backbone_indices), \
            f"The max value ({max(backbone_indices)}) of backbone_indices should be " \
            f"less than the length of feat_channels ({len(backbone.feat_channels)})."
        self.backbone = backbone

        assert len(backbone_indices) > 1, "The lenght of backbone_indices " \
            "should be greater than 1"
        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]

        # head
        if len(arm_out_chs) == 1:
            arm_out_chs = arm_out_chs * len(backbone_indices)
        assert len(arm_out_chs) == len(backbone_indices), "The length of " \
            "arm_out_chs and backbone_indices should be equal"

        self.ppseg_head = PPLiteSegHead(backbone_out_chs, arm_out_chs,
                                        cm_bin_sizes, cm_out_ch, arm_type,
                                        resize_mode)

        if len(seg_head_inter_chs) == 1:
            seg_head_inter_chs = seg_head_inter_chs * len(backbone_indices)
        assert len(seg_head_inter_chs) == len(backbone_indices), "The length of " \
            "seg_head_inter_chs and backbone_indices should be equal"
        self.seg_heads = nn.LayerList()  # [..., head_16, head32]
        for in_ch, mid_ch in zip(arm_out_chs, seg_head_inter_chs):
            self.seg_heads.append(SegHead(in_ch, mid_ch, num_classes))

        # pretrained
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]

        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        assert len(feats_backbone) >= len(self.backbone_indices), \
            f"The nums of backbone feats ({len(feats_backbone)}) should be greater or " \
            f"equal than the nums of backbone_indices ({len(self.backbone_indices)})"

        feats_selected = [feats_backbone[i] for i in self.backbone_indices]

        feats_head = self.ppseg_head(feats_selected)  # [..., x8, x16, x32]

        if self.training:
            logit_list = []

            for x, seg_head in zip(feats_head, self.seg_heads):
                x = seg_head(x)
                logit_list.append(x)

            logit_list = [
                F.interpolate(
                    x, x_hw, mode='bilinear', align_corners=False)
                for x in logit_list
            ]
        else:
            x = self.seg_heads[0](feats_head[0])
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
            logit_list = [x]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class PPLiteSegHead(nn.Module):
    """
    The head of PPLiteSeg.

    Args:
        backbone_out_chs (List(Tensor)): The channels of output tensors in the backbone.
        arm_out_chs (List(int)): The out channels of each arm module.
        cm_bin_sizes (List(int)): The bin size of context module.
        cm_out_ch (int): The output channel of the last context module.
        arm_type (str): The type of attention refinement module.
        resize_mode (str): The resize mode for the upsampling operation in decoder.
    """

    def __init__(self, backbone_out_chs, arm_out_chs, cm_bin_sizes, cm_out_ch,
                 arm_type, resize_mode):
        super().__init__()

        self.cm = PPContextModule(backbone_out_chs[-1], cm_out_ch, cm_out_ch,
                                  cm_bin_sizes)

        assert hasattr(layers,arm_type), \
            "Not support arm_type ({})".format(arm_type)
        arm_class = eval("layers." + arm_type)

        self.arm_list = nn.LayerList()  # [..., arm8, arm16, arm32]
        for i in range(len(backbone_out_chs)):
            low_chs = backbone_out_chs[i]
            high_ch = cm_out_ch if i == len(
                backbone_out_chs) - 1 else arm_out_chs[i + 1]
            out_ch = arm_out_chs[i]
            arm = arm_class(
                low_chs, high_ch, out_ch, ksize=3, resize_mode=resize_mode)
            self.arm_list.append(arm)

    def forward(self, in_feat_list):
        """
        Args:
            in_feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
        Returns:
            out_feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
                The length of in_feat_list and out_feat_list are the same.
        """

        high_feat = self.cm(in_feat_list[-1])
        out_feat_list = []

        for i in reversed(range(len(in_feat_list))):
            low_feat = in_feat_list[i]
            arm = self.arm_list[i]
            high_feat = arm(low_feat, high_feat)
            out_feat_list.insert(0, high_feat)

        return out_feat_list


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
