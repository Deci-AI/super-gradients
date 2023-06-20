"""
Repvgg Pytorch Implementation. This model trains a vgg with residual blocks
but during inference (in deployment mode) will convert the model to vgg model.
Pretrained models: https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq
Refrerences:
    [1] https://github.com/DingXiaoH/RepVGG
    [2] https://arxiv.org/pdf/2101.03697.pdf

Based on https://github.com/DingXiaoH/RepVGG
"""
from typing import Union

import torch.nn as nn

from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.modules import RepVGGBlock, SEBlock
from super_gradients.training.models.sg_module import SgModule
from super_gradients.modules.repvgg_block import fuse_repvgg_blocks_residual_branches
from super_gradients.training.utils.utils import get_param


class RepVGG(SgModule):
    def __init__(
        self,
        struct,
        num_classes=1000,
        width_multiplier=None,
        build_residual_branches=True,
        use_se=False,
        backbone_mode=False,
        in_channels=3,
    ):
        """
        :param struct: list containing number of blocks per repvgg stage
        :param num_classes: number of classes if nut in backbone mode
        :param width_multiplier: list of per stage width multiplier or float if using single value for all stages
        :param build_residual_branches: whether to add residual connections or not
        :param use_se: use squeeze and excitation layers
        :param backbone_mode: if true, dropping the final linear layer
        :param in_channels: input channels
        """
        super(RepVGG, self).__init__()

        if isinstance(width_multiplier, float):
            width_multiplier = [width_multiplier] * 4
        else:
            assert len(width_multiplier) == 4

        self.build_residual_branches = build_residual_branches
        self.use_se = use_se
        self.backbone_mode = backbone_mode

        self.in_planes = int(64 * width_multiplier[0])

        self.stem = RepVGGBlock(
            in_channels=in_channels,
            out_channels=self.in_planes,
            stride=2,
            build_residual_branches=build_residual_branches,
            activation_type=nn.ReLU,
            activation_kwargs=dict(inplace=True),
            se_type=SEBlock if self.use_se else nn.Identity,
            se_kwargs=dict(in_channels=self.in_planes, internal_neurons=self.in_planes // 16) if self.use_se else None,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), struct[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), struct[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), struct[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), struct[3], stride=2)
        if not self.backbone_mode:
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
            self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

        if not build_residual_branches:
            self.eval()  # fusing has to be made in eval mode. When called in init, model will be built in eval mode
            fuse_repvgg_blocks_residual_branches(self)

        self.final_width_mult = width_multiplier[3]

    def _make_stage(self, planes, struct, stride):
        strides = [stride] + [1] * (struct - 1)
        blocks = []
        for stride in strides:
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    stride=stride,
                    groups=1,
                    build_residual_branches=self.build_residual_branches,
                    activation_type=nn.ReLU,
                    activation_kwargs=dict(inplace=True),
                    se_type=SEBlock if self.use_se else nn.Identity,
                    se_kwargs=dict(in_channels=self.in_planes, internal_neurons=self.in_planes // 16) if self.use_se else None,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        if not self.backbone_mode:
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        if self.build_residual_branches:
            fuse_repvgg_blocks_residual_branches(self)

    def train(self, mode: bool = True):

        assert (
            not mode or self.build_residual_branches
        ), "Trying to train a model without residual branches, set arch_params.build_residual_branches to True and retrain the model"
        super(RepVGG, self).train(mode=mode)

    def replace_head(self, new_num_classes=None, new_head=None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_head is not None:
            self.linear = new_head
        else:
            self.linear = nn.Linear(int(512 * self.final_width_mult), new_num_classes)


@register_model(Models.REPVGG_CUSTOM)
class RepVggCustom(RepVGG):
    def __init__(self, arch_params):
        super().__init__(
            struct=arch_params.struct,
            num_classes=arch_params.num_classes,
            width_multiplier=arch_params.width_multiplier,
            build_residual_branches=arch_params.build_residual_branches,
            use_se=get_param(arch_params, "use_se", False),
            backbone_mode=get_param(arch_params, "backbone_mode", False),
            in_channels=get_param(arch_params, "in_channels", 3),
        )


@register_model(Models.REPVGG_A0)
class RepVggA0(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5])
        super().__init__(arch_params=arch_params)


@register_model(Models.REPVGG_A1)
class RepVggA1(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[2, 4, 14, 1], width_multiplier=[1, 1, 1, 2.5])
        super().__init__(arch_params=arch_params)


@register_model(Models.REPVGG_A2)
class RepVggA2(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[2, 4, 14, 1], width_multiplier=[1.5, 1.5, 1.5, 2.75])
        super().__init__(arch_params=arch_params)


@register_model(Models.REPVGG_B0)
class RepVggB0(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[4, 6, 16, 1], width_multiplier=[1, 1, 1, 2.5])
        super().__init__(arch_params=arch_params)


@register_model(Models.REPVGG_B1)
class RepVggB1(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4])
        super().__init__(arch_params=arch_params)


@register_model(Models.REPVGG_B2)
class RepVggB2(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5])
        super().__init__(arch_params=arch_params)


@register_model(Models.REPVGG_B3)
class RepVggB3(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5])
        super().__init__(arch_params=arch_params)


@register_model(Models.REPVGG_D2SE)
class RepVggD2SE(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[8, 14, 24, 1], width_multiplier=[2.5, 2.5, 2.5, 5])
        super().__init__(arch_params=arch_params)
