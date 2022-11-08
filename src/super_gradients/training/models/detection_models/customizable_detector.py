"""
A base for a detection network built according to the following scheme:
 * constructed from nested arch_params;
 * inside arch_params each nested level (module) has an explicit type and its required parameters
 * each module accepts in_channels and other parameters
 * each module defines out_channels property on construction
"""


from typing import Union, List

import torch
from torch import nn
from omegaconf import DictConfig

from super_gradients.training.utils.utils import HpmStruct, get_param
from super_gradients.training.models.sg_module import SgModule
from super_gradients.common.factories import DetectionModulesFactory
from super_gradients.modules.detection_modules import BaseDetectionModule
from super_gradients.common.registry import register_detection_module


@register_detection_module("NStageBackbone")
class NStageBackbone(BaseDetectionModule):
    """
    A backbone with a stem -> N stages -> context module
    Returns outputs of the layers listed in out_layers
    """

    def __init__(
        self,
        in_channels: int,
        out_layers: List[str],
        stem: Union[str, HpmStruct, DictConfig],
        stages: Union[str, HpmStruct, DictConfig],
        context_module: Union[str, HpmStruct, DictConfig],
    ):
        """
        :param out_layers: names of layers to output from the following options: 'stem', 'stageN', 'context_module'
        """
        super().__init__(in_channels)
        factory = DetectionModulesFactory()

        self.num_stages = len(stages)
        self.stem = factory.get(factory.insert_module_param(stem, "in_channels", in_channels))
        prev_channels = self.stem.out_channels
        for i in range(self.num_stages):
            new_stage = factory.get(factory.insert_module_param(stages[i], "in_channels", prev_channels))
            setattr(self, f"stage{i + 1}", new_stage)
            prev_channels = new_stage.out_channels
        self.context_module = factory.get(factory.get(factory.insert_module_param(context_module, "in_channels", prev_channels)))

        self.out_layers = out_layers
        self._out_channels = self._define_out_channels()

    def _define_out_channels(self):
        out_channels = []
        for layer in self.out_layers:
            out_channels.append(getattr(self, layer).out_channels)
        return out_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):

        outputs = []
        all_layers = ["stem"] + [f"stage{i}" for i in range(1, self.num_stages + 1)] + ["context_module"]
        for layer in all_layers:
            x = getattr(self, layer)(x)
            if layer in self.out_layers:
                outputs.append(x)

        return outputs


@register_detection_module("PANNeck")
class PANNeck(BaseDetectionModule):
    """
    A PAN (path aggregation network) neck with 4 stages (2 up-sampling and 2 down-sampling stages)
    Returns outputs of neck stage 2, stage 3, stage 4
    """

    def __init__(
        self,
        in_channels: List[int],
        neck1: Union[str, HpmStruct, DictConfig],
        neck2: Union[str, HpmStruct, DictConfig],
        neck3: Union[str, HpmStruct, DictConfig],
        neck4: Union[str, HpmStruct, DictConfig],
    ):
        super().__init__(in_channels)
        c3_out_channels, c4_out_channels, c5_out_channels = in_channels

        factory = DetectionModulesFactory()
        self.neck1 = factory.get(factory.insert_module_param(neck1, "in_channels", [c5_out_channels, c4_out_channels]))
        self.neck2 = factory.get(factory.insert_module_param(neck2, "in_channels", [self.neck1.out_channels[1], c3_out_channels]))
        self.neck3 = factory.get(factory.insert_module_param(neck3, "in_channels", [self.neck2.out_channels[1], self.neck2.out_channels[0]]))
        self.neck4 = factory.get(factory.insert_module_param(neck4, "in_channels", [self.neck3.out_channels, self.neck1.out_channels[0]]))

        self._out_channels = [
            self.neck2.out_channels[1],
            self.neck3.out_channels,
            self.neck4.out_channels,
        ]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        c3, c4, c5 = inputs

        x_n1_inter, x = self.neck1([c5, c4])
        x_n2_inter, p3 = self.neck2([x, c3])
        p4 = self.neck3([p3, x_n2_inter])
        p5 = self.neck4([p4, x_n1_inter])

        return p3, p4, p5


@register_detection_module("NHeads")
class NHeads(BaseDetectionModule):
    """
    Apply N heads in parallel and combine predictions into the shape expected by SG detection losses
    """

    def __init__(self, in_channels: List[int], num_classes: int, heads_list: Union[str, HpmStruct, DictConfig]):
        super().__init__(in_channels)
        factory = DetectionModulesFactory()
        heads_list = self._pass_num_classes(heads_list, factory, num_classes)

        self.num_heads = len(heads_list)
        for i in range(self.num_heads):
            new_head = factory.get(factory.insert_module_param(heads_list[i], "in_channels", in_channels[i]))
            setattr(self, f"head{i + 1}", new_head)

    @staticmethod
    def _pass_num_classes(heads_list, factory, num_classes):
        for i in range(len(heads_list)):
            heads_list[i] = factory.insert_module_param(heads_list[i], "num_classes", num_classes)
        return heads_list

    @property
    def out_channels(self):
        return None

    def forward(self, inputs):
        outputs = []
        for i in range(self.num_heads):
            outputs.append(getattr(self, f"head{i + 1}")(inputs[i]))

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
    Each submodule with its parameters must be defined explicitly.
    Modules should follow the interface of BaseDetectionModule
    """

    def __init__(self, arch_params: Union[HpmStruct, DictConfig], in_channels: int = 3):
        """
        :param type_mapping: can be passed to resolve string type names in arch_params to actual types
        """
        super().__init__()

        factory = DetectionModulesFactory()

        # move num_classes into heads params
        if get_param(arch_params, "num_classes"):
            arch_params.heads = factory.insert_module_param(arch_params.heads, "num_classes", arch_params.num_classes)

        self.arch_params = arch_params
        self.backbone = factory.get(factory.insert_module_param(arch_params.backbone, "in_channels", in_channels))
        self.neck = factory.get(factory.insert_module_param(arch_params.neck, "in_channels", self.backbone.out_channels))
        self.heads = factory.get(factory.insert_module_param(arch_params.heads, "in_channels", self.neck.out_channels))

        self._initialize_weights(arch_params)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.heads(x)

    def _initialize_weights(self, arch_params: Union[HpmStruct, DictConfig]):

        bn_eps = get_param(arch_params, "bn_eps", None)
        bn_momentum = get_param(arch_params, "bn_momentum", None)
        inplace_act = get_param(arch_params, "inplace_act", True)

        for m in self.modules():
            t = type(m)
            if t is nn.BatchNorm2d:
                m.eps = bn_eps if bn_eps else m.eps
                m.momentum = bn_momentum if bn_momentum else m.momentum
            elif inplace_act and t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, nn.Mish]:
                m.inplace = True

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        for module in self.modules():
            if module != self and hasattr(module, "prep_model_for_conversion"):
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
