"""
A base for a detection network built according to the following scheme:
 * constructed from nested arch_params;
 * inside arch_params each nested level (module) has an explicit type parameter and other parameters it requires
 * each module accepts arch_params and in_channels
 * each module defines out_channels attribute on construction
 * in_channels defines channels of tensor(s) that will be accepted by a module in forward
 * out_channels defines channels of tensor(s) that will be returned by a module  in forward
"""


from typing import Union, List, Dict, Type

import torch
from torch import nn
from omegaconf import DictConfig

from super_gradients.training.utils.utils import HpmStruct, get_param
from super_gradients.training.models.sg_module import SgModule
from super_gradients.common.factories import DetectionModulesFactory


class NStageBackbone(nn.Module):
    """
    A backbone with a stem -> N stages -> context module
    Returns outputs of the layers listed in arch_params.out_layers
    """
    def __init__(self, arch_params: Union[HpmStruct, DictConfig], in_channels: int):
        super().__init__()
        factory = arch_params.factory

        self.num_stages = max([int(k.replace('stage', '')) for k in arch_params.keys() if k.startswith('stage')])
        self.stem = factory.get(arch_params.stem, in_channels)
        prev_channels = self.stem.out_channels
        for i in range(1, self.num_stages + 1):
            setattr(self, f'stage{i}', factory.get(arch_params[f'stage{i}'], prev_channels))
            prev_channels = getattr(self, f'stage{i}').out_channels
        self.context_module = factory.get(arch_params.context_module, self.stage4.out_channels)

        self.out_layers = arch_params.out_layers
        self.out_channels = self._get_out_channels()

    def _get_out_channels(self):
        out_channels = []
        for layer in self.out_layers:
            out_channels.append(getattr(self, layer).out_channels)
        return out_channels

    def forward(self, x):

        outputs = []
        all_layers = ['stem'] + [f'stage{i}' for i in range(1, self.num_stages + 1)] + ['context_module']
        for layer in all_layers:
            x = getattr(self, layer)(x)
            if layer in self.out_layers:
                outputs.append(x)

        return outputs


class PANNeck(nn.Module):
    """
    A PAN (path aggregation network) neck with 4 stages (2 up-sampling and 2 down-sampling stages)
    Returns outputs of neck stage 2, stage 3, stage 4
    """
    def __init__(self, arch_params: Union[HpmStruct, DictConfig], in_channels: List[int]):
        super().__init__()
        c3_out_channels, c4_out_channels, c5_out_channels = in_channels

        factory = arch_params.factory
        self.neck1 = factory.get(arch_params.neck1, [c5_out_channels, c4_out_channels])
        self.neck2 = factory.get(arch_params.neck2, [self.neck1.out_channels[1], c3_out_channels])
        self.neck3 = factory.get(arch_params.neck3, [self.neck2.out_channels[1], self.neck2.out_channels[0]])
        self.neck4 = factory.get(arch_params.neck4, [self.neck3.out_channels, self.neck1.out_channels[0]])

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


class NHeads(nn.Module):
    """
    Apply N heads in parallel and combine predictions into the shape expected by SG detection losses
    """
    def __init__(self, arch_params: Union[HpmStruct, DictConfig], in_channels: List[int]):
        super().__init__()
        arch_params = self._pass_num_classes(arch_params)
        factory = arch_params.factory

        self.num_heads = max([int(k.replace('head', '')) for k in arch_params.keys() if k.startswith('head')])
        for i in range(self.num_heads):
            setattr(self, f'head{i + 1}', factory.get(arch_params[f'head{i + 1}'], in_channels[i]))

    @staticmethod
    def _pass_num_classes(arch_params: HpmStruct):
        for i in range(3):
            arch_params[f'head{i + 1}'].num_classes = arch_params.num_classes
        return arch_params

    def forward(self, inputs):
        outputs = []
        for i in range(self.num_heads):
            outputs.append(getattr(self, f'head{i + 1}')(inputs[i]))

        return self.combine_preds(outputs)

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

    By default, initializes BatchNorm eps to 1e-3, momentum to 0.03 and sets all activations to be inplace
    """

    def __init__(self, arch_params: Union[HpmStruct, DictConfig], type_mapping: Dict[str, Type], in_channels: int = 3):
        """
        :param type_mapping: can be passed to resolve string type names in arch_params to actual types
        """
        super().__init__()

        self.factory = DetectionModulesFactory(type_mapping)
        self.arch_params = arch_params
        self.backbone = self.factory.get(arch_params.backbone, in_channels)
        self.neck = self.factory.get(arch_params.neck, self.backbone.out_channels)
        self.heads = self.factory.get(arch_params.heads, self.neck.out_channels)

        self._initialize_weights(arch_params)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.heads(x)

    def _initialize_weights(self, arch_params: Union[HpmStruct, DictConfig]):

        bn_eps = get_param(arch_params, 'bn_eps', 1e-05)
        bn_momentum = get_param(arch_params, 'bn_momentum', 0.1)
        inplace_act = get_param(arch_params, 'inplace_act', True)

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

    def replace_head(self, new_num_classes: int = None, new_head: nn.Module = None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_head is not None:
            self.heads = new_head
        else:
            self.arch_params.heads.num_classes = new_num_classes
            self.heads = self.factory.get(self.arch_params.heads, self.neck.out_channels)
            self._initialize_weights(self.arch_params)
