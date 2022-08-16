import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, List, Tuple
from super_gradients.training.utils.module_utils import ConvBNReLU, UpsampleMode, make_upsample_module
from super_gradients.training.models.segmentation_models.stdc import SegmentationHead


class PPLiteSegBase(nn.Module):
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
                 backbone,
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
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

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
        self.proj_skip = ConvBNReLU(skip_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.up_x = make_upsample_module(scale_factor=2, upsample_mode=upsample_mode, align_corners=align_corners)
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


if __name__ == '__main__':
    m = UAFM(in_channels=128, skip_channels=1024, out_channels=96)
    x = torch.randn(2, 128, 1024 // 32, 2048 // 32)
    y = torch.randn(2, 1024, 1024 // 16, 2048 // 16)

    print(m(x, y).shape)

    torch.onnx.export(m, (x, y), "/Users/liork/Downloads/tmp.onnx", opset_version=11)
