import torch
import torch.nn as nn
from typing import Union, List
from super_gradients.modules import ConvBNReLU

from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.modules.sampling import make_upsample_module
from super_gradients.common.data_types.enum.upsample_mode import UpsampleMode
from super_gradients.training.models.segmentation_models.stdc import AbstractSTDCBackbone, STDC1Backbone, STDC2Backbone
from super_gradients.training.models.segmentation_models.common import SegmentationHead
from super_gradients.training.models.segmentation_models.segmentation_module import SegmentationModule
from super_gradients.training.utils import HpmStruct, get_param, torch_version_is_greater_or_equal
from super_gradients.training.models.segmentation_models.context_modules import SPPM


class UAFM(nn.Module):
    """
    Unified Attention Fusion Module, which uses mean and max values across the spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        up_factor: int,
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.BILINEAR,
        align_corners: bool = False,
    ):
        """
        :params in_channels: num_channels of input feature map.
        :param skip_channels: num_channels of skip connection feature map.
        :param out_channels: num out channels after features fusion.
        :param up_factor: upsample scale factor of the input feature map.
        :param upsample_mode: see UpsampleMode for valid options.
        """
        super().__init__()
        self.conv_atten = nn.Sequential(
            ConvBNReLU(4, 2, kernel_size=3, padding=1, bias=False), ConvBNReLU(2, 1, kernel_size=3, padding=1, bias=False, use_activation=False)
        )

        self.proj_skip = nn.Identity() if skip_channels == in_channels else ConvBNReLU(skip_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.up_x = nn.Identity() if up_factor == 1 else make_upsample_module(scale_factor=up_factor, upsample_mode=upsample_mode, align_corners=align_corners)
        self.conv_out = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x, skip):
        """
        :param x: input feature map to upsample before fusion.
        :param skip: skip connection feature map.
        """
        x = self.up_x(x)
        skip = self.proj_skip(skip)

        atten = torch.cat([*self._avg_max_spatial_reduce(x, use_concat=False), *self._avg_max_spatial_reduce(skip, use_concat=False)], dim=1)
        atten = self.conv_atten(atten)
        atten = torch.sigmoid(atten)

        out = x * atten + skip * (1 - atten)
        out = self.conv_out(out)
        return out

    @staticmethod
    def _avg_max_spatial_reduce(x, use_concat: bool = False):
        reduced = [torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]]
        if use_concat:
            reduced = torch.cat(reduced, dim=1)
        return reduced


class PPLiteSegEncoder(nn.Module):
    """
    Encoder for PPLiteSeg, include backbone followed by a context module.
    """

    def __init__(self, backbone: AbstractSTDCBackbone, projection_channels_list: List[int], context_module: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.context_module = context_module
        feats_channels = backbone.get_backbone_output_number_of_channels()
        self.proj_convs = nn.ModuleList(
            [ConvBNReLU(feat_ch, proj_ch, kernel_size=3, padding=1, bias=False) for feat_ch, proj_ch in zip(feats_channels, projection_channels_list)]
        )
        self.projection_channels_list = projection_channels_list

    def get_output_number_of_channels(self) -> List[int]:
        channels_list = self.projection_channels_list
        if hasattr(self.context_module, "out_channels"):
            channels_list.append(self.context_module.out_channels)
        return channels_list

    def forward(self, x):
        feats = self.backbone(x)
        y = self.context_module(feats[-1])
        feats = [conv(f) for conv, f in zip(self.proj_convs, feats)]
        return feats + [y]


class PPLiteSegDecoder(nn.Module):
    """
    PPLiteSegDecoder using UAFM blocks to fuse feature maps.
    """

    def __init__(self, encoder_channels: List[int], up_factors: List[int], out_channels: List[int], upsample_mode, align_corners: bool):
        super().__init__()
        # Make a copy of channels list, to prevent out of scope changes.
        encoder_channels = encoder_channels.copy()
        encoder_channels.reverse()
        in_channels = encoder_channels.pop(0)

        # TODO - assert argument length
        self.up_stages = nn.ModuleList()
        for skip_ch, up_factor, out_ch in zip(encoder_channels, up_factors, out_channels):
            self.up_stages.append(
                UAFM(
                    in_channels=in_channels,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    up_factor=up_factor,
                    upsample_mode=upsample_mode,
                    align_corners=align_corners,
                )
            )
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

    def __init__(
        self,
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
        head_scale_factor: int,
        head_upsample_mode: Union[UpsampleMode, str],
        head_mid_channels: int,
        dropout: float,
        use_aux_heads: bool,
        aux_hidden_channels: List[int],
        aux_scale_factors: List[int],
    ):
        """
        :param backbone: Backbone nn.Module should implement the abstract class `AbstractSTDCBackbone`.
        :param projection_channels_list: channels list to project encoder features before fusing with the decoder
            stream.
        :param sppm_inter_channels: num channels in each sppm pooling branch.
        :param sppm_out_channels: The number of output channels after sppm module.
        :param sppm_pool_sizes: spatial output sizes of the pooled feature maps.
        :param sppm_upsample_mode: Upsample mode to original size after pooling.
        :param decoder_up_factors: list upsample factor per decoder stage.
        :param decoder_channels: list of num_channels per decoder stage.
        :param decoder_upsample_mode: upsample mode in decoder stages, see UpsampleMode for valid options.
        :param head_scale_factor: scale factor for final the segmentation head logits.
        :param head_upsample_mode: upsample mode to final prediction sizes, see UpsampleMode for valid options.
        :param head_mid_channels: num of hidden channels in segmentation head.
        :param use_aux_heads: set True when training, output extra Auxiliary feature maps from the encoder module.
        :param aux_hidden_channels: List of hidden channels in auxiliary segmentation heads.
        :param aux_scale_factors: list of uppsample factors for final auxiliary heads logits.
        """
        super().__init__(use_aux_heads=use_aux_heads)

        # Init Encoder
        backbone_out_channels = backbone.get_backbone_output_number_of_channels()
        assert len(backbone_out_channels) == len(projection_channels_list), (
            f"The length of backbone outputs ({backbone_out_channels}) should match the length of projection channels" f"({len(projection_channels_list)})."
        )
        context = SPPM(
            in_channels=backbone_out_channels[-1],
            inter_channels=sppm_inter_channels,
            out_channels=sppm_out_channels,
            pool_sizes=sppm_pool_sizes,
            upsample_mode=sppm_upsample_mode,
            align_corners=align_corners,
        )
        self.encoder = PPLiteSegEncoder(backbone=backbone, context_module=context, projection_channels_list=projection_channels_list)
        encoder_channels = self.encoder.get_output_number_of_channels()

        # Init Decoder
        self.decoder = PPLiteSegDecoder(
            encoder_channels=encoder_channels,
            up_factors=decoder_up_factors,
            out_channels=decoder_channels,
            upsample_mode=decoder_upsample_mode,
            align_corners=align_corners,
        )

        # Init Segmentation classification heads
        self.seg_head = nn.Sequential(
            SegmentationHead(in_channels=decoder_channels[-1], mid_channels=head_mid_channels, num_classes=num_classes, dropout=dropout),
            make_upsample_module(scale_factor=head_scale_factor, upsample_mode=head_upsample_mode, align_corners=align_corners),
        )
        # Auxiliary heads
        if self.use_aux_heads:
            encoder_out_channels = projection_channels_list
            self.aux_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        SegmentationHead(backbone_ch, hidden_ch, num_classes, dropout=dropout),
                        make_upsample_module(scale_factor=scale_factor, upsample_mode=head_upsample_mode, align_corners=align_corners),
                    )
                    for backbone_ch, hidden_ch, scale_factor in zip(encoder_out_channels, aux_hidden_channels, aux_scale_factors)
                ]
            )
        self.init_params()

    def _remove_auxiliary_heads(self):
        if hasattr(self, "aux_heads"):
            del self.aux_heads

    @property
    def backbone(self) -> nn.Module:
        """
        Support SG load backbone when training.
        """
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

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """
        Custom param groups for training:
            - Different lr for backbone and the rest, if `multiply_head_lr` key is in `training_params`.
        """
        multiply_head_lr = get_param(training_params, "multiply_head_lr", 1)
        multiply_lr_params, no_multiply_params = self._separate_lr_multiply_params()
        param_groups = [
            {"named_params": no_multiply_params, "lr": lr, "name": "no_multiply_params"},
            {"named_params": multiply_lr_params, "lr": lr * multiply_head_lr, "name": "multiply_lr_params"},
        ]
        return param_groups

    def update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int, training_params: HpmStruct, total_batch: int) -> list:
        multiply_head_lr = get_param(training_params, "multiply_head_lr", 1)
        for param_group in param_groups:
            param_group["lr"] = lr
            if param_group["name"] == "multiply_lr_params":
                param_group["lr"] *= multiply_head_lr
        return param_groups

    def _separate_lr_multiply_params(self):
        """
        Separate backbone params from the rest.
        :return: iterators of groups named_parameters.
        """
        multiply_lr_params, no_multiply_params = {}, {}
        for name, param in self.named_parameters():
            if "encoder.backbone" in name:
                no_multiply_params[name] = param
            else:
                multiply_lr_params[name] = param
        return multiply_lr_params.items(), no_multiply_params.items()

    def prep_model_for_conversion(self, input_size: Union[tuple, list], stride_ratio: int = 32, **kwargs):
        if not torch_version_is_greater_or_equal(1, 11):
            raise RuntimeError("PPLiteSeg model ONNX export requires torch => 1.11, torch installed: " + str(torch.__version__))
        super().prep_model_for_conversion(input_size, **kwargs)
        if isinstance(self.encoder.context_module, SPPM):
            self.encoder.context_module.prep_model_for_conversion(input_size=input_size, stride_ratio=stride_ratio)

    def replace_head(self, new_num_classes: int, **kwargs):
        for module in self.modules():
            if isinstance(module, SegmentationHead):
                module.replace_num_classes(new_num_classes)


@register_model(Models.PP_LITE_B_SEG)
@register_model(Models.PP_LITE_B_SEG50)
@register_model(Models.PP_LITE_B_SEG75)
class PPLiteSegB(PPLiteSegBase):
    def __init__(self, arch_params: HpmStruct):
        backbone = STDC2Backbone(in_channels=get_param(arch_params, "in_channels", 3), out_down_ratios=[8, 16, 32])
        super().__init__(
            num_classes=get_param(arch_params, "num_classes"),
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
            head_scale_factor=8,
            head_upsample_mode="bilinear",
            head_mid_channels=64,
            dropout=get_param(arch_params, "dropout", 0.0),
            use_aux_heads=get_param(arch_params, "use_aux_heads", False),
            aux_hidden_channels=[32, 64, 64],
            aux_scale_factors=[8, 16, 32],
        )


@register_model(Models.PP_LITE_T_SEG)
@register_model(Models.PP_LITE_T_SEG50)
@register_model(Models.PP_LITE_T_SEG75)
class PPLiteSegT(PPLiteSegBase):
    def __init__(self, arch_params: HpmStruct):
        backbone = STDC1Backbone(in_channels=get_param(arch_params, "in_channels", 3), out_down_ratios=[8, 16, 32])
        super().__init__(
            num_classes=get_param(arch_params, "num_classes"),
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
            head_scale_factor=8,
            head_upsample_mode="bilinear",
            head_mid_channels=32,
            dropout=get_param(arch_params, "dropout", 0.0),
            use_aux_heads=get_param(arch_params, "use_aux_heads", False),
            aux_hidden_channels=[32, 64, 64],
            aux_scale_factors=[8, 16, 32],
        )
