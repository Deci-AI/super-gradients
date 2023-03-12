import torch.nn as nn
from typing import Optional, Union, List

from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.training.utils import HpmStruct, get_param
from super_gradients.training import models
from super_gradients.training.models.segmentation_models.segmentation_module import SegmentationModule
from super_gradients.modules.sampling import UpsampleMode
from super_gradients.modules.sampling import make_upsample_module
from super_gradients.modules.repvgg_block import fuse_repvgg_blocks_residual_branches
from super_gradients.training.models.segmentation_models.unet.unet_encoder import UNetBackboneBase, Encoder
from super_gradients.training.models.segmentation_models.context_modules import AbstractContextModule
from super_gradients.training.models.segmentation_models.unet.unet_decoder import Decoder
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.context_modules_factory import ContextModulesFactory
from super_gradients.training.models.segmentation_models.common import SegmentationHead


class UNetBase(SegmentationModule):
    @resolve_param("context_module", ContextModulesFactory())
    def __init__(
        self,
        num_classes: int,
        use_aux_heads: bool,
        final_upsample_factor: int,
        head_hidden_channels: Optional[int],
        head_upsample_mode: Union[UpsampleMode, str],
        align_corners: bool,
        backbone_params: dict,
        context_module: AbstractContextModule,
        decoder_params: dict,
        aux_heads_params: dict,
        dropout: float,
    ):
        """
        :param num_classes: num classes to predict.
        :param use_aux_heads: Whether to use auxiliary heads.
        :param final_upsample_factor: Final upsample scale factor after the segmentation head.
        :param head_hidden_channels: num channels before the last classification layer. see `mid_channels` in
            `SegmentationHead` class.
        :param head_upsample_mode: UpsampleMode of segmentation and auxiliary heads.
        :param align_corners: align_corners arg of segmentation and auxiliary heads.
        :param backbone_params: params to build a `UNetBackboneBase`, include the following keys:
            - strides_list: List[int], list of stride per stage.
            - width_list: List[int], list of num channels per stage.
            - num_blocks_list: List[int], list of num blocks per stage.
            - block_types_list: List[Union[DownBlockType, int]], list of block types per stage.
            - is_out_feature_list: List[bool], list of flags whether stage features should be an output.
            - in_channels: int, num channels of the input to the backbone module.
            - block_params: dict, argument to be passed to the block types constructors. i.e for `RegnetXStage`
                block_params should include bottleneck_ratio, group_width and se_ratio.
        :param decoder_params: params to build a `Decoder`, include the following keys:
            - up_block_repeat_list: List[int], num of blocks per decoder stage, the `block` implementation depends on
                the up-block type.
            - skip_expansion: float, skip expansion ratio value, before fusing the skip features from the encoder with
                the decoder features, a projection convolution is applied upon the encoder features to project the
                num_channels by skip_expansion.
            - decoder_scale: float, num_channels width ratio between encoder stages and decoder stages.
            - up_blocks: List[Type[AbstractUpFuseBlock]], list of AbstractUpFuseBlock types.
            - is_skip_list: List[bool], List of flags whether to use feature-map from encoder stage as skip connection
                or not.
        :param aux_heads_params: params to initiate auxiliary heads, include the following keys:
            - use_aux_list: List[bool], whether to append to auxiliary head per encoder stage.
            - aux_heads_factor: List[int], Upsample factor per encoder stage.
            - aux_hidden_channels: List[int], Hidden num channels before last classification layer, per encoder stage.
            - aux_out_channels: List[int], Output channels, can be refers as num_classes, of auxiliary head per encoder
                stage.
        :param dropout: dropout probability of segmentation and auxiliary heads.
        """
        super().__init__(use_aux_heads=use_aux_heads)
        self.num_classes = num_classes
        # Init Backbone
        backbone = UNetBackboneBase(**backbone_params)
        # Init Encoder
        self.encoder = Encoder(backbone, context_module)
        # Init Decoder
        self.decoder = Decoder(skip_channels_list=self.encoder.get_output_number_of_channels(), **decoder_params)
        # Init Segmentation Head
        self.seg_head = nn.Sequential(
            SegmentationHead(
                in_channels=self.decoder.up_channels_list[-1],
                mid_channels=head_hidden_channels or self.decoder.up_channels_list[-1],
                num_classes=self.num_classes,
                dropout=dropout,
            ),
            nn.Identity()
            if final_upsample_factor == 1
            else make_upsample_module(scale_factor=final_upsample_factor, upsample_mode=head_upsample_mode, align_corners=align_corners),
        )
        # Init Aux Heads
        if self.use_aux_heads:
            # Aux heads are applied if both conditions are true, use_aux_list is set as True and the correspondent
            # backbone features are outputted and set as True in backbone is_out_feature_list.
            aux_heads_params["use_aux_list"] = [a and b for a, b in zip(aux_heads_params["use_aux_list"], backbone_params["is_out_feature_list"])]
            self.aux_heads = self.init_aux_heads(
                in_channels_list=self.encoder.get_all_number_of_channels(),
                upsample_mode=head_upsample_mode,
                align_corners=align_corners,
                dropout=dropout,
                **aux_heads_params,
            )
            self.use_aux_feats = [a and b for a, b in zip(aux_heads_params["use_aux_list"], backbone_params["is_out_feature_list"]) if b]
        self.init_params()

    @staticmethod
    def init_aux_heads(
        in_channels_list: List[int],
        use_aux_list: List[bool],
        aux_heads_factor: List[int],
        aux_hidden_channels: List[int],
        aux_out_channels: List[int],
        dropout: float,
        upsample_mode: Union[str, UpsampleMode],
        align_corners: Optional[bool] = None,
    ):
        """
        :param use_aux_list: whether to append to auxiliary head per encoder stage.
        :param in_channels_list: list of input channels to the auxiliary segmentation heads.
        :param aux_heads_factor: list of upsample scale factors to apply at the end of the auxiliary segmentation heads.
        :param aux_hidden_channels: list of segmentation heads hidden channels.
        :param aux_out_channels: list of segmentation heads out channels, usually set as num_classes or 1 for detail
            edge heads.
        :param dropout: dropout probability factor.
        :param upsample_mode: see UpsampleMode for supported options.
        :return: nn.ModuleList
        """
        heads = nn.ModuleList(
            [
                nn.Sequential(
                    SegmentationHead(ch, hid_ch, out_ch, dropout=dropout),
                    make_upsample_module(scale_factor=scale, upsample_mode=upsample_mode, align_corners=align_corners),
                )
                for ch, scale, hid_ch, out_ch, use_aux in zip(in_channels_list, aux_heads_factor, aux_hidden_channels, aux_out_channels, use_aux_list)
                if use_aux
            ]
        )
        return heads

    def forward(self, x):
        encoder_feats = self.encoder(x)
        x = self.decoder(encoder_feats)
        x = self.seg_head(x)
        if not self.use_aux_heads:
            return x
        encoder_feats = [f for i, f in enumerate(encoder_feats) if self.use_aux_feats[i]]
        aux_feats = [aux_head(feat) for feat, aux_head in zip(encoder_feats[-len(self.aux_heads) :], self.aux_heads)]
        aux_feats.reverse()
        return tuple([x] + aux_feats)

    def _remove_auxiliary_heads(self):
        if hasattr(self, "aux_heads"):
            del self.aux_heads

    @property
    def backbone(self):
        return self.encoder

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """
        Custom param groups for training:
            - Different lr for head and rest, if `multiply_head_lr` key is in `training_params`.
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
        multiply_lr_params, no_multiply_params = {}, {}
        for name, param in self.named_parameters():
            if "backbone." in name:
                no_multiply_params[name] = param
            else:
                multiply_lr_params[name] = param
        return multiply_lr_params.items(), no_multiply_params.items()

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, append_sigmoid: bool = False, append_softmax: bool = False, **kwargs):
        super().prep_model_for_conversion(input_size=input_size, **kwargs)
        fuse_repvgg_blocks_residual_branches(self)
        if append_sigmoid:
            self.seg_head.add_module("sigmoid", nn.Sigmoid())
        if append_softmax:
            self.seg_head.add_module("softmax", nn.Softmax(dim=1))

    def replace_head(self, new_num_classes: int, **kwargs):
        for module in self.modules():
            if isinstance(module, SegmentationHead):
                module.replace_num_classes(new_num_classes)


@register_model(Models.UNET_CUSTOM)
class UNetCustom(UNetBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params = HpmStruct(**models.get_arch_params("unet_default_arch_params.yaml", overriding_params=arch_params.to_dict()))
        super().__init__(
            num_classes=get_param(arch_params, "num_classes"),
            use_aux_heads=get_param(arch_params, "use_aux_heads", False),
            final_upsample_factor=get_param(arch_params, "final_upsample_factor", 1),
            head_hidden_channels=get_param(arch_params, "head_hidden_channels"),
            head_upsample_mode=get_param(arch_params, "head_upsample_mode", UpsampleMode.BILINEAR),
            align_corners=get_param(arch_params, "align_corners", False),
            backbone_params=get_param(arch_params, "backbone_params"),
            context_module=get_param(arch_params, "context_module", nn.Identity()),
            decoder_params=get_param(arch_params, "decoder_params"),
            aux_heads_params=get_param(arch_params, "aux_heads_params"),
            dropout=get_param(arch_params, "dropout", 0.0),
        )


@register_model(Models.UNET)
class UNet(UNetCustom):
    """
    implementation of:
     "U-Net: Convolutional Networks for Biomedical Image Segmentation", https://arxiv.org/pdf/1505.04597.pdf
    The upsample operation is done by using bilinear interpolation which is reported to show better results.
    """

    def __init__(self, arch_params: HpmStruct):
        arch_params = HpmStruct(**models.get_arch_params("unet_arch_params.yaml", arch_params.to_dict()))
        super().__init__(arch_params)
