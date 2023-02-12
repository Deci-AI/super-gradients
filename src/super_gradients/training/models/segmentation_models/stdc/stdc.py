"""
Implementation of paper: "Rethinking BiSeNet For Real-time Semantic Segmentation", https://arxiv.org/abs/2104.13188
Based on original implementation: https://github.com/MichaelFan01/STDC-Seg, cloned 23/08/2021, commit 59ff37f
"""
from typing import Union, Optional
import torch.nn as nn

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training.models import SgModule
from super_gradients.training.models.segmentation_models.registry import SEGMENTATION_BACKBONES
from super_gradients.training.models.segmentation_models.stdc.stdc_decoder import STDCDecoder
from super_gradients.training.models.segmentation_models.stdc.stdc_encoder import STDCBackbone, STDC1Backbone, STDC2Backbone
from super_gradients.training.utils import get_param, HpmStruct
from super_gradients.modules import ConvBNReLU
from super_gradients.training.models.segmentation_models.common import SegmentationHead, AbstractSegmentationBackbone

# default STDC argument as paper.
STDC_SEG_DEFAULT_ARGS = {"context_fuse_channels": 128, "ffm_channels": 256, "aux_head_channels": 64, "detail_head_channels": 64}


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


class STDCClassification(STDCClassificationBase):
    def __init__(self, arch_params: HpmStruct):
        super().__init__(
            backbone=get_param(arch_params, "backbone"), num_classes=get_param(arch_params, "num_classes"), dropout=get_param(arch_params, "dropout", 0.2)
        )


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

    @resolve_param("backbone", BaseFactory(SEGMENTATION_BACKBONES))
    @resolve_param("initial_upsample", TransformsFactory())
    @resolve_param("final_upsample", TransformsFactory())
    def __init__(
        self,
        backbone: AbstractSegmentationBackbone,
        num_classes: int,
        initial_upsample: Optional[nn.Module],
        final_upsample: Optional[nn.Module],
        context_fuse_channels: int,
        ffm_channels: int,
        aux_head_channels: int,
        detail_head_channels: int,
        use_aux_heads: bool,
        dropout: float,
    ):
        super(STDCSegmentationBase, self).__init__()
        self.backbone = backbone
        self._validate_backbone()
        self._use_aux_heads = use_aux_heads
        self.initial_upsample = initial_upsample or nn.Identity()

        skip_channels = [_.channels for _ in backbone.get_backbone_output_spec()]
        stage3_s8_channels, stage4_s16_channels, stage5_s32_channels = skip_channels

        self.decoder = STDCDecoder(
            skip_channels_list=skip_channels, context_fuse_channels=context_fuse_channels, ffm_channels=ffm_channels, ffm_projection_channels=None
        )

        self.segmentation_head = nn.Sequential(SegmentationHead(ffm_channels, ffm_channels, num_classes, dropout=dropout), final_upsample or nn.Identity())

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

    def _validate_backbone(self):
        n_outputs = len(self.backbone.get_backbone_output_spec())
        if n_outputs != 3:
            raise AssertionError(
                f"{self.__class__.__name__} assumes 3 outputs from the backbone. The provided {self.backbone.__class__.__name__} has {n_outputs}."
            )

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        """
        Prepare model for conversion, force use_aux_heads mode False and delete auxiliary and detail heads. Replace
        ContextEmbeddingOnline which cause compilation issues and not supported in some compilations,
        to ContextEmbeddingFixedSize.
        """
        # set to false and delete auxiliary and detail heads modules.
        self.use_aux_heads = False
        self.decoder.prep_model_for_conversion(input_size, **kwargs)

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
        self._use_aux_heads = use_aux

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
        x = self.initial_upsample(x)
        feat8, feat16, feat32 = self.backbone(x)
        feat_out = self.decoder([feat8, feat16, feat32])
        feat_out = self.segmentation_head(feat_out)

        if not self.use_aux_heads:
            return feat_out

        detail_out8 = self.detail_head8(feat8)
        aux_out_s16 = self.aux_head_s16(feat16)
        aux_out_s32 = self.aux_head_s32(feat32)

        return feat_out, aux_out_s32, aux_out_s16, detail_out8

    def replace_head(self, new_num_classes: int, **kwargs):
        ffm_channels = self.decoder.ffm.attention_block[-2].out_channels
        dropout = self.segmentation_head[0].seg_head[1].p

        # Output layer's replacement- first modules in the sequences are the SegmentationHead modules.
        self.segmentation_head[0] = SegmentationHead(ffm_channels, ffm_channels, new_num_classes, dropout=dropout)
        if self.use_aux_heads:
            stage3_s8_channels, stage4_s16_channels, stage5_s32_channels = [_.channels for _ in self.backbone.get_backbone_output_spec()]
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
            if "decoder." in name:
                no_multiply_params[name] = param
            else:
                multiply_lr_params[name] = param
        return multiply_lr_params.items(), no_multiply_params.items()


class CustomSTDCSegmentation(STDCSegmentationBase):
    """
    Fully customized STDC Segmentation factory module.
    """

    def __init__(self, arch_params: HpmStruct):
        super().__init__(
            backbone=get_param(arch_params, "backbone"),
            initial_upsample=get_param(arch_params, "initial_upsample"),
            final_upsample=get_param(arch_params, "final_upsample"),
            num_classes=get_param(arch_params, "num_classes"),
            context_fuse_channels=get_param(arch_params, "context_fuse_channels", 128),
            ffm_channels=get_param(arch_params, "ffm_channels", 256),
            aux_head_channels=get_param(arch_params, "aux_head_channels", 64),
            detail_head_channels=get_param(arch_params, "detail_head_channels", 64),
            use_aux_heads=get_param(arch_params, "use_aux_heads", True),
            dropout=get_param(arch_params, "dropout", 0.2),
        )


class STDC1Classification(STDCClassification):
    def __init__(self, arch_params: HpmStruct):
        backbone = STDC1Backbone(in_channels=get_param(arch_params, "input_channels", 3), out_down_ratios=(32,))
        arch_params.override(**{"backbone": backbone})
        super().__init__(arch_params)


class STDC2Classification(STDCClassification):
    def __init__(self, arch_params: HpmStruct):
        backbone = STDC2Backbone(in_channels=get_param(arch_params, "input_channels", 3), out_down_ratios=(32,))
        arch_params.override(**{"backbone": backbone})
        super().__init__(arch_params)


class STDC1Seg(CustomSTDCSegmentation):
    def __init__(self, arch_params: HpmStruct):
        backbone = STDC1Backbone(in_channels=get_param(arch_params, "in_channels", 3), out_down_ratios=[8, 16, 32])

        custom_params = {"backbone": backbone, **STDC_SEG_DEFAULT_ARGS}
        arch_params.override(**custom_params)
        super().__init__(arch_params)


class STDC2Seg(CustomSTDCSegmentation):
    def __init__(self, arch_params: HpmStruct):
        backbone = STDC2Backbone(in_channels=get_param(arch_params, "in_channels", 3), out_down_ratios=[8, 16, 32])

        custom_params = {"backbone": backbone, **STDC_SEG_DEFAULT_ARGS}
        arch_params.override(**custom_params)
        super().__init__(arch_params)
