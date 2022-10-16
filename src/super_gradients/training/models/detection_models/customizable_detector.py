"""
A base for a detection network built according to the following scheme:
 * constructed from nested arch_params;
 * inside arch_params each nested level (module) has an explicit type parameter and other parameters it requires
 * each module accepts arch_params and in_channels
 * each module defines out_channels attribute on construction
 * in_channels defines channels of tensor(s) that will be accepted by a module in forward
 * out_channels defines channels of tensor(s) that will be returned by a module  in forward
"""


from typing import Union, List, Dict

import torch
from torch import nn
from omegaconf import DictConfig

from super_gradients.training.utils.utils import HpmStruct, get_param
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.utils.hydra_utils import recursive_type_name_to_type


class FourStageBackbone(nn.Module):
    """
    A backbone with a stem -> 4 stages -> context module
    Returns outputs of stage 2, stage 3, context module
    """
    def __init__(self, arch_params: Union[HpmStruct, DictConfig], in_channels: int):
        super().__init__()

        StemCls = arch_params.stem.type
        Stage1Cls = arch_params.stage1.type
        Stage2Cls = arch_params.stage2.type
        Stage3Cls = arch_params.stage3.type
        Stage4Cls = arch_params.stage4.type
        ContextCls = arch_params.context_module.type

        self.stem = StemCls(arch_params.stem, in_channels)
        self.stage1 = Stage1Cls(arch_params.stage1, self.stem.out_channels)
        self.stage2 = Stage2Cls(arch_params.stage2, self.stage1.out_channels)
        self.stage3 = Stage3Cls(arch_params.stage3, self.stage2.out_channels)
        self.stage4 = Stage4Cls(arch_params.stage4, self.stage3.out_channels)
        self.context_module = ContextCls(arch_params.context_module, self.stage4.out_channels)

        self.out_channels = [
            self.stage2.out_channels,
            self.stage3.out_channels,
            self.context_module.out_channels
        ]

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.stage1(c1)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5_ = self.stage4(c4)
        c5 = self.context_module(c5_)

        return c3, c4, c5


class PANNeck(nn.Module):
    """
    A PAN (path aggregation network) neck with 4 stages (2 up-sampling and 2 down-sampling stages)
    Returns outputs of neck stage 2, stage 3, stage 4
    """
    def __init__(self, arch_params: Union[HpmStruct, DictConfig], in_channels: List[int]):
        super().__init__()
        c3_out_channels, c4_out_channels, c5_out_channels = in_channels

        self.neck1 = arch_params.neck1.type(arch_params.neck1, [c5_out_channels, c4_out_channels])
        self.neck2 = arch_params.neck2.type(arch_params.neck2, [self.neck1.out_channels[1], c3_out_channels])
        self.neck3 = arch_params.neck3.type(arch_params.neck3, [self.neck2.out_channels[1], self.neck2.out_channels[0]])
        self.neck4 = arch_params.neck4.type(arch_params.neck4, [self.neck3.out_channels, self.neck1.out_channels[0]])

        self.out_channels = [
            self.neck2.out_channels[1],
            self.neck3.out_channels,
            self.neck4.out_channels,
        ]

    def forward(self, inputs):
        c3, c4, c5 = inputs

        x_n1_inter, x = self.neck1([c5, c4])
        x_n2_inter, p3 = self.neck2([x, c3])
        p4 = self.neck3([p3, x_n2_inter])
        p5 = self.neck4([p4, x_n1_inter])

        return p3, p4, p5


class ThreeHeads(nn.Module):
    """
    Apply three heads and combine predictions into the shape expected by SG detection losses
    """
    def __init__(self, arch_params: Union[HpmStruct, DictConfig], in_channels: List[int]):
        super().__init__()
        self.head1 = arch_params.head1.type(arch_params.head1, in_channels[0])
        self.head2 = arch_params.head2.type(arch_params.head2, in_channels[1])
        self.head3 = arch_params.head3.type(arch_params.head3, in_channels[2])

    def forward(self, inputs):
        p3, p4, p5 = inputs
        return self.combine_preds([self.head1(p3), self.head2(p4), self.head3(p5)])

    def combine_preds(self, preds):
        outputs = []
        outputs_logits = []
        for output, output_logits in preds:
            outputs.append(output)
            outputs_logits.append(output_logits)

        return outputs if self.training else (torch.cat(outputs, 1), outputs_logits)


class CustomizableDetector(SgModule):
    """
    A customizable detector with backbone -> neck -> heads

    A type of each submodule must be defined explicitly. The defaults are:
      * backbone - FourStageBackbone
      * neck - PANNeck
      * heads - ThreeHeads

    By default initializes BatchNorm eps to 1e-3, momentum to 0.03 and sets all activations to be inplace
    """

    def __init__(self, arch_params: Union[HpmStruct, DictConfig], type_mapping: Dict = None):
        """
        :param arch_params:
        :param type_mapping: can be passed to recursively resolve string type names in arch_params to actual types
        """
        super().__init__()
        if type_mapping is not None:
            recursive_type_name_to_type(arch_params, type_mapping)

        self.arch_params = arch_params
        BackboneCls = get_param(arch_params.backbone, 'type', FourStageBackbone)
        NeckCls = get_param(arch_params.neck, 'type', PANNeck)
        HeadsCls = get_param(arch_params.heads, 'type', ThreeHeads)

        self.backbone = BackboneCls(arch_params.backbone, 3)
        self.neck = NeckCls(arch_params.neck, self.backbone.out_channels)
        self.heads = HeadsCls(arch_params.heads, self.neck.out_channels)

        self.initialize_weights(get_param(arch_params, 'bn_eps', 1e-3),
                                get_param(arch_params, 'bn_momentum', 0.03),
                                get_param(arch_params, 'inplace_act', True))

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.heads(x)

    def initialize_weights(self, bn_eps: float, bn_momentum: float, inplace_act: bool):
        for m in self.modules():
            t = type(m)
            if t is nn.BatchNorm2d:
                m.eps = bn_eps
                m.momentum = bn_momentum
            elif inplace_act and t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, nn.Mish]:
                m.inplace = True

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        for module in self.modules():
            if module != self and hasattr(module, 'prep_model_for_conversion'):
                module.prep_model_for_conversion(input_size, **kwargs)
