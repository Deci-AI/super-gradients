"""
Implementation of paper: "Rethinking BiSeNet For Real-time Semantic Segmentation", https://arxiv.org/abs/2104.13188
Based on original implementation: https://github.com/MichaelFan01/STDC-Seg, cloned 23/08/2021, commit 59ff37f
"""
from typing import Union, List
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.models import SgModule
from super_gradients.training.utils import get_param, HpmStruct
from super_gradients.modules import ConvBNReLU, Residual
from super_gradients.training.models.segmentation_models.common import SegmentationHead


# default STDC argument as paper.
STDC_SEG_DEFAULT_ARGS = {"context_fuse_channels": 128, "ffm_channels": 256, "aux_head_channels": 64, "detail_head_channels": 64}


class STDCBlock(nn.Module):
    """
    STDC building block, known as Short Term Dense Concatenate module.
    In STDC module, the kernel size of first block is 1, and the rest of them are simply set as 3.
    """

    def __init__(self, in_channels: int, out_channels: int, steps: int, stdc_downsample_mode: str, stride: int):
        """
        :param steps: The total number of convs in this module, 1 conv 1x1 and (steps - 1) conv3x3.
        :param stdc_downsample_mode: downsample mode in stdc block, supported `avg_pool` for average-pooling and
         `dw_conv` for depthwise-convolution.
        """
        super().__init__()
        if steps not in [2, 3, 4]:
            raise ValueError(f"only 2, 3, 4 steps number are supported, found: {steps}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.steps = steps
        self.stdc_downsample_mode = stdc_downsample_mode
        self.stride = stride
        self.conv_list = nn.ModuleList()
        # build first step conv 1x1.
        self.conv_list.append(ConvBNReLU(in_channels, out_channels // 2, kernel_size=1, bias=False))
        # build skip connection after first convolution.
        if stride == 1:
            self.skip_step1 = Residual()
        elif stdc_downsample_mode == "avg_pool":
            self.skip_step1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif stdc_downsample_mode == "dw_conv":
            self.skip_step1 = ConvBNReLU(
                out_channels // 2, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False, groups=out_channels // 2, use_activation=False
            )
        else:
            raise ValueError(f"stdc_downsample mode is not supported: found {stdc_downsample_mode}," f" must be in [avg_pool, dw_conv]")

        in_channels = out_channels // 2
        mid_channels = in_channels
        # build rest conv3x3 layers.
        for idx in range(1, steps):
            if idx < steps - 1:
                mid_channels //= 2
            conv = ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_list.append(conv)
            in_channels = mid_channels

        # add dw conv before second step for down sample if stride = 2.
        if stride == 2:
            self.conv_list[1] = nn.Sequential(
                ConvBNReLU(
                    out_channels // 2, out_channels // 2, kernel_size=3, stride=2, padding=1, groups=out_channels // 2, use_activation=False, bias=False
                ),
                self.conv_list[1],
            )

    def forward(self, x):
        out_list = []
        # run first conv
        x = self.conv_list[0](x)
        out_list.append(self.skip_step1(x))

        for conv in self.conv_list[1:]:
            x = conv(x)
            out_list.append(x)

        out = torch.cat(out_list, dim=1)
        return out


class AbstractSTDCBackbone(nn.Module, ABC):
    """
    All backbones for STDC segmentation models must implement this class.
    """

    def validate_backbone(self):
        if len(self.get_backbone_output_number_of_channels()) != 3:
            raise ValueError(f"Backbone for STDC segmentation must output 3 feature maps," f" found: {len(self.get_backbone_output_number_of_channels())}.")

    @abstractmethod
    def get_backbone_output_number_of_channels(self) -> List[int]:
        """
        :return: list on stages num channels.
        """
        raise NotImplementedError()


class STDCBackbone(AbstractSTDCBackbone):
    def __init__(
        self,
        block_types: list,
        ch_widths: list,
        num_blocks: list,
        stdc_steps: int = 4,
        stdc_downsample_mode: str = "avg_pool",
        in_channels: int = 3,
        out_down_ratios: Union[tuple, list] = (32,),
    ):
        """
        :param block_types: list of block type for each stage, supported `conv` for ConvBNRelu with 3x3 kernel.
        :param ch_widths: list of output num of channels for each stage.
        :param num_blocks: list of the number of repeating blocks in each stage.
        :param stdc_steps: num of convs steps in each block.
        :param stdc_downsample_mode: downsample mode in stdc block, supported `avg_pool` for average-pooling and
         `dw_conv` for depthwise-convolution.
        :param in_channels: num channels of the input image.
        :param out_down_ratios: down ratio of output feature maps required from the backbone,
            default (32,) for classification.
        """
        super(STDCBackbone, self).__init__()
        if not (len(block_types) == len(ch_widths) == len(num_blocks)):
            raise ValueError(
                f"STDC architecture configuration, block_types, ch_widths, num_blocks, must be defined for the same number"
                f" of stages, found: {len(block_types)} for block_type, {len(ch_widths)} for ch_widths, "
                f"{len(num_blocks)} for num_blocks"
            )

        self.out_widths = []
        self.stages = nn.ModuleDict()
        self.out_stage_keys = []
        down_ratio = 2
        for block_type, width, blocks in zip(block_types, ch_widths, num_blocks):
            block_name = f"block_s{down_ratio}"
            self.stages[block_name] = self._make_stage(
                in_channels=in_channels,
                out_channels=width,
                block_type=block_type,
                num_blocks=blocks,
                stdc_steps=stdc_steps,
                stdc_downsample_mode=stdc_downsample_mode,
            )
            if down_ratio in out_down_ratios:
                self.out_stage_keys.append(block_name)
                self.out_widths.append(width)
            in_channels = width
            down_ratio *= 2

    def _make_stage(self, in_channels: int, out_channels: int, block_type: str, num_blocks: int, stdc_downsample_mode: str, stdc_steps: int = 4):
        """
        :param in_channels: input channels of stage.
        :param out_channels: output channels of stage.
        :param block_type: stage building block, supported `conv` for 3x3 ConvBNRelu, or `stdc` for STDCBlock.
        :param num_blocks: num of blocks in each stage.
        :param stdc_steps: number of conv3x3 steps in each STDC block, referred as `num blocks` in paper.
        :param stdc_downsample_mode: downsample mode in stdc block, supported `avg_pool` for average-pooling and
         `dw_conv` for depthwise-convolution.
        :return: nn.Module
        """
        if block_type == "conv":
            block = ConvBNReLU
            kwargs = {"kernel_size": 3, "padding": 1, "bias": False}
        elif block_type == "stdc":
            block = STDCBlock
            kwargs = {"steps": stdc_steps, "stdc_downsample_mode": stdc_downsample_mode}
        else:
            raise ValueError(f"Block type not supported: {block_type}, excepted: `conv` or `stdc`")

        # first block to apply stride 2.
        blocks = nn.ModuleList([block(in_channels, out_channels, stride=2, **kwargs)])
        # build rest of blocks
        for i in range(num_blocks - 1):
            blocks.append(block(out_channels, out_channels, stride=1, **kwargs))

        return nn.Sequential(*blocks)

    def forward(self, x):
        outputs = []
        for stage_name, stage in self.stages.items():
            x = stage(x)
            if stage_name in self.out_stage_keys:
                outputs.append(x)
        return tuple(outputs)

    def get_backbone_output_number_of_channels(self) -> List[int]:
        return self.out_widths


class STDCClassificationBase(SgModule):
    """
    Base module for classification model based on STDCs backbones
    """

    def __init__(self, backbone: STDCBackbone, num_classes: int, dropout: float):
        super(STDCClassificationBase, self).__init__()
        self.backbone = backbone
        last_channels = self.backbone.out_widths[-1]
        head_channels = max(1024, last_channels)

        self.conv_last = ConvBNReLU(last_channels, head_channels, 1, 1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(head_channels, head_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(head_channels, num_classes, bias=False)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.backbone(x)[-1]
        # original implementation, why to use power?
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear(out)
        return out


@register_model(Models.STDC_CUSTOM_CLS)
class STDCClassification(STDCClassificationBase):
    def __init__(self, arch_params: HpmStruct):
        super().__init__(
            backbone=get_param(arch_params, "backbone"), num_classes=get_param(arch_params, "num_classes"), dropout=get_param(arch_params, "dropout", 0.2)
        )


class AttentionRefinementModule(nn.Module):
    """
    AttentionRefinementModule to apply on the last two backbone stages.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(AttentionRefinementModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
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
    """

    def __init__(self, spatial_channels: int, context_channels: int, out_channels: int):
        super(FeatureFusionModule, self).__init__()
        self.spatial_channels = spatial_channels
        self.context_channels = context_channels
        self.out_channels = out_channels

        self.pw_conv = ConvBNReLU(spatial_channels + context_channels, out_channels, kernel_size=1, stride=1, bias=False)
        # TODO - used without bias in convolutions by mistake, try to reproduce with bias=True
        self.attention_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_channels=out_channels, out_channels=out_channels // 4, kernel_size=1, use_normalization=False, bias=False),
            nn.Conv2d(in_channels=out_channels // 4, out_channels=out_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, spatial_feats, context_feats):
        feat = torch.cat([spatial_feats, context_feats], dim=1)
        feat = self.pw_conv(feat)
        atten = self.attention_block(feat)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class ContextEmbedding(nn.Module):
    """
    ContextEmbedding module that use global average pooling to 1x1 to extract context information, and then upsample
    to original input size.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ContextEmbedding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_embedding = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        self.fixed_size = False

    def forward(self, x):
        out_height, out_width = x.size()[2:]
        x = self.context_embedding(x)
        return F.interpolate(x, size=(out_height, out_width), mode="nearest")

    def to_fixed_size(self, upsample_size: Union[list, tuple]):
        if self.fixed_size:
            return
        self.fixed_size = True

        self.context_embedding.add_module("upsample", nn.Upsample(scale_factor=upsample_size, mode="nearest"))

        self.forward = self.context_embedding.forward


class ContextPath(nn.Module):
    """
    ContextPath in STDC output both the Spatial path and Context path. This module include a STDCBackbone and output
    the stage3 feature map with down_ratio = 8 as the spatial feature map, and context feature map which is a result of
    upsampling and fusion of context embedding, stage5 and stage4 after Arm modules, Which is also with same resolution
    of the spatial feature map, down_ration = 8.
    :param backbone: Backbone of type AbstractSTDCBackbone that return info about backbone output channels.
    :param fuse_channels: num channels of the fused context path.
    :param use_aux_heads: set True when training, output extra Auxiliary feature maps of the two last stages of the
     backbone.
    """

    def __init__(self, backbone: AbstractSTDCBackbone, fuse_channels: int, use_aux_heads: bool):
        super(ContextPath, self).__init__()

        self.fuse_channels = fuse_channels
        self.use_aux_heads = use_aux_heads

        self.backbone = backbone
        # get num of channels for two last stages
        channels16, channels32 = self.backbone.get_backbone_output_number_of_channels()[-2:]

        self.context_embedding = ContextEmbedding(channels32, fuse_channels)

        self.arm32 = AttentionRefinementModule(channels32, fuse_channels)
        self.upsample32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), ConvBNReLU(fuse_channels, fuse_channels, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.arm16 = AttentionRefinementModule(channels16, fuse_channels)
        self.upsample16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), ConvBNReLU(fuse_channels, fuse_channels, kernel_size=3, padding=1, stride=1, bias=False)
        )

    def forward(self, x):
        feat8, feat16, feat32 = self.backbone(x)

        ce_feats = self.context_embedding(feat32)
        feat32_arm = self.arm32(feat32)
        feat32_arm = feat32_arm + ce_feats

        feat32_up = self.upsample32(feat32_arm)

        feat16_arm = self.arm16(feat16)
        feat16_arm = feat16_arm + feat32_up
        feat16_up = self.upsample16(feat16_arm)

        if self.use_aux_heads:
            return feat8, feat16_up, feat16, feat32
        return feat8, feat16_up

    def prep_for_conversion(self, input_size):
        if input_size[-2] % 32 != 0 or input_size[-1] % 32 != 0:
            raise ValueError(f"Expected image dimensions to be divisible by 32, got {input_size[-2]}x{input_size[-1]}")

        context_embedding_up_size = (input_size[-2] // 32, input_size[-1] // 32)
        self.context_embedding.to_fixed_size(context_embedding_up_size)


class STDCSegmentationBase(SgModule):
    """
    Base STDC Segmentation Module.
    :param backbone: Backbone of type AbstractSTDCBackbone that return info about backbone output channels.
    :param num_classes: num of dataset classes, exclude ignore label.
    :param context_fuse_channels: num of output channels in ContextPath ARM feature fusion.
    :param ffm_channels: num of output channels of Feature Fusion Module.
    :param aux_head_channels: Num of hidden channels in Auxiliary segmentation heads.
    :param detail_head_channels: Num of hidden channels in Detail segmentation heads.
    :param use_aux_heads: set True when training, attach Auxiliary and Detail heads. For compilation / inference mode
        set False.
    :param dropout: segmentation heads dropout.
    """

    @resolve_param("backbone", BaseFactory({"STDCBackbone": STDCBackbone}))
    def __init__(
        self,
        backbone: AbstractSTDCBackbone,
        num_classes: int,
        context_fuse_channels: int,
        ffm_channels: int,
        aux_head_channels: int,
        detail_head_channels: int,
        use_aux_heads: bool,
        dropout: float,
    ):
        super(STDCSegmentationBase, self).__init__()
        backbone.validate_backbone()
        self._use_aux_heads = use_aux_heads

        self.cp = ContextPath(backbone, context_fuse_channels, use_aux_heads=use_aux_heads)

        stage3_s8_channels, stage4_s16_channels, stage5_s32_channels = backbone.get_backbone_output_number_of_channels()

        self.ffm = FeatureFusionModule(spatial_channels=stage3_s8_channels, context_channels=context_fuse_channels, out_channels=ffm_channels)
        # Main segmentation head
        self.segmentation_head = nn.Sequential(
            SegmentationHead(ffm_channels, ffm_channels, num_classes, dropout=dropout), nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        )

        if self._use_aux_heads:
            # Auxiliary heads
            self.aux_head_s16 = nn.Sequential(
                SegmentationHead(stage4_s16_channels, aux_head_channels, num_classes, dropout=dropout),
                nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
            )
            self.aux_head_s32 = nn.Sequential(
                SegmentationHead(stage5_s32_channels, aux_head_channels, num_classes, dropout=dropout),
                nn.Upsample(scale_factor=32, mode="bilinear", align_corners=True),
            )
            # Detail head
            self.detail_head8 = nn.Sequential(
                SegmentationHead(stage3_s8_channels, detail_head_channels, 1, dropout=dropout), nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
            )

        self.init_params()

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        """
        Prepare model for conversion, force use_aux_heads mode False and delete auxiliary and detail heads. Replace
        ContextEmbeddingOnline which cause compilation issues and not supported in some compilations,
        to ContextEmbeddingFixedSize.
        """
        # set to false and delete auxiliary and detail heads modules.
        self.use_aux_heads = False

        self.cp.prep_for_conversion(input_size)

    def _remove_auxiliary_and_detail_heads(self):
        attributes_to_delete = ["aux_head_s16", "aux_head_s32", "detail_head8"]
        for attr in attributes_to_delete:
            if hasattr(self, attr):
                delattr(self, attr)

    @property
    def use_aux_heads(self):
        return self._use_aux_heads

    @use_aux_heads.setter
    def use_aux_heads(self, use_aux: bool):
        """
        private setter for self._use_aux_heads, called every time an assignment to self._use_aux_heads is applied.
        if use_aux is False, `_remove_auxiliary_and_detail_heads` is called to delete auxiliary and detail heads.
        if use_aux is True, and self._use_aux_heads was already set to False a ValueError is raised, recreating
            aux and detail heads outside init method is not allowed, and the module should be recreated.
        """
        if use_aux is True and self._use_aux_heads is False:
            raise ValueError("Cant turn use_aux_heads from False to True, you should initiate the module again with" " `use_aux_heads=True`")
        if not use_aux:
            self._remove_auxiliary_and_detail_heads()
        self.cp.use_aux_heads = use_aux
        self._use_aux_heads = use_aux

    @property
    def backbone(self):
        """
        For Trainer load_backbone compatibility.
        """
        return self.cp.backbone

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        cp_outs = self.cp(x)
        feat8, feat_cp8 = cp_outs[0], cp_outs[1]
        # fuse stage 3 with result of context path after ARM modules.
        feat_out = self.ffm(spatial_feats=feat8, context_feats=feat_cp8)
        feat_out = self.segmentation_head(feat_out)

        if not self.use_aux_heads:
            return feat_out
        feat16, feat32 = cp_outs[2], cp_outs[3]
        detail_out8 = self.detail_head8(feat8)

        aux_out_s16 = self.aux_head_s16(feat16)
        aux_out_s32 = self.aux_head_s32(feat32)

        return feat_out, aux_out_s32, aux_out_s16, detail_out8

    def replace_head(self, new_num_classes: int, **kwargs):
        ffm_channels = self.ffm.attention_block[-2].out_channels
        dropout = self.segmentation_head[0].seg_head[1].p

        # Output layer's replacement- first modules in the sequences are the SegmentationHead modules.
        self.segmentation_head[0] = SegmentationHead(ffm_channels, ffm_channels, new_num_classes, dropout=dropout)
        if self.use_aux_heads:
            stage3_s8_channels, stage4_s16_channels, stage5_s32_channels = self.backbone.get_backbone_output_number_of_channels()
            aux_head_channels = self.aux_head_s16[0].seg_head[-1].in_channels
            detail_head_channels = self.detail_head8[0].seg_head[-1].in_channels

            self.aux_head_s16[0] = SegmentationHead(stage4_s16_channels, aux_head_channels, new_num_classes, dropout=dropout)

            self.aux_head_s32[0] = SegmentationHead(stage5_s32_channels, aux_head_channels, new_num_classes, dropout=dropout)
            # Detail head
            self.detail_head8[0] = SegmentationHead(stage3_s8_channels, detail_head_channels, 1, dropout=dropout)

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """
        Custom param groups for STDC training:
            - Different lr for context path and heads, if `multiply_head_lr` key is in `training_params`.
            - Add extra Detail loss params to optimizer.
        """

        extra_train_params = training_params.loss.get_train_named_params() if hasattr(training_params.loss, "get_train_named_params") else None
        multiply_head_lr = get_param(training_params, "multiply_head_lr", 1)

        multiply_lr_params, no_multiply_params = self._separate_lr_multiply_params()
        param_groups = [
            {"named_params": no_multiply_params, "lr": lr, "name": "no_multiply_params"},
            {"named_params": multiply_lr_params, "lr": lr * multiply_head_lr, "name": "multiply_lr_params"},
        ]

        if extra_train_params is not None:
            param_groups.append({"named_params": extra_train_params, "lr": lr, "weight_decay": 0.0, "name": "detail_params"})

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
        Separate ContextPath params from the rest.
        :return: iterators of groups named_parameters.
        """
        multiply_lr_params, no_multiply_params = {}, {}
        for name, param in self.named_parameters():
            if "cp." in name:
                no_multiply_params[name] = param
            else:
                multiply_lr_params[name] = param
        return multiply_lr_params.items(), no_multiply_params.items()


@register_model(Models.STDC_CUSTOM)
@register_model("custom_stdc")  # deprecated naming convention. will be dropped in v4
class CustomSTDCSegmentation(STDCSegmentationBase):
    """
    Fully customized STDC Segmentation factory module.
    """

    def __init__(self, arch_params: HpmStruct):
        super().__init__(
            backbone=get_param(arch_params, "backbone"),
            num_classes=get_param(arch_params, "num_classes"),
            context_fuse_channels=get_param(arch_params, "context_fuse_channels", 128),
            ffm_channels=get_param(arch_params, "ffm_channels", 256),
            aux_head_channels=get_param(arch_params, "aux_head_channels", 64),
            detail_head_channels=get_param(arch_params, "detail_head_channels", 64),
            use_aux_heads=get_param(arch_params, "use_aux_heads", True),
            dropout=get_param(arch_params, "dropout", 0.2),
        )


class STDC1Backbone(STDCBackbone):
    def __init__(self, in_channels: int = 3, out_down_ratios: Union[tuple, list] = (32,)):
        super().__init__(
            block_types=["conv", "conv", "stdc", "stdc", "stdc"],
            ch_widths=[32, 64, 256, 512, 1024],
            num_blocks=[1, 1, 2, 2, 2],
            stdc_steps=4,
            in_channels=in_channels,
            out_down_ratios=out_down_ratios,
        )


class STDC2Backbone(STDCBackbone):
    def __init__(self, in_channels: int = 3, out_down_ratios: Union[tuple, list] = (32,)):
        super().__init__(
            block_types=["conv", "conv", "stdc", "stdc", "stdc"],
            ch_widths=[32, 64, 256, 512, 1024],
            num_blocks=[1, 1, 4, 5, 3],
            stdc_steps=4,
            in_channels=in_channels,
            out_down_ratios=out_down_ratios,
        )


@register_model(Models.STDC1_CLASSIFICATION)
class STDC1Classification(STDCClassification):
    def __init__(self, arch_params: HpmStruct):
        backbone = STDC1Backbone(in_channels=get_param(arch_params, "input_channels", 3), out_down_ratios=(32,))
        arch_params.override(**{"backbone": backbone})
        super().__init__(arch_params)


@register_model(Models.STDC2_CLASSIFICATION)
class STDC2Classification(STDCClassification):
    def __init__(self, arch_params: HpmStruct):
        backbone = STDC2Backbone(in_channels=get_param(arch_params, "input_channels", 3), out_down_ratios=(32,))
        arch_params.override(**{"backbone": backbone})
        super().__init__(arch_params)


@register_model(Models.STDC1_SEG)
@register_model(Models.STDC1_SEG50)
@register_model(Models.STDC1_SEG75)
class STDC1Seg(CustomSTDCSegmentation):
    def __init__(self, arch_params: HpmStruct):
        backbone = STDC1Backbone(in_channels=get_param(arch_params, "in_channels", 3), out_down_ratios=[8, 16, 32])

        custom_params = {"backbone": backbone, **STDC_SEG_DEFAULT_ARGS}
        arch_params.override(**custom_params)
        super().__init__(arch_params)


@register_model(Models.STDC2_SEG)
@register_model(Models.STDC2_SEG50)
@register_model(Models.STDC2_SEG75)
class STDC2Seg(CustomSTDCSegmentation):
    def __init__(self, arch_params: HpmStruct):
        backbone = STDC2Backbone(in_channels=get_param(arch_params, "in_channels", 3), out_down_ratios=[8, 16, 32])

        custom_params = {"backbone": backbone, **STDC_SEG_DEFAULT_ARGS}
        arch_params.override(**custom_params)
        super().__init__(arch_params)
