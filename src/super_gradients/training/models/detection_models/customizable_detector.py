"""
A base for a detection network built according to the following scheme:
 * constructed from nested arch_params;
 * inside arch_params each nested level (module) has an explicit type and its required parameters
 * each module accepts in_channels and other parameters
 * each module defines out_channels property on construction
"""


from typing import Union

from torch import nn
from omegaconf import DictConfig

from super_gradients.training.utils.utils import HpmStruct, get_param
from super_gradients.training.models.sg_module import SgModule
import super_gradients.common.factories.detection_modules_factory as det_factory


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

        factory = det_factory.DetectionModulesFactory()

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
            factory = det_factory.DetectionModulesFactory()
            self.arch_params.heads = factory.insert_module_param(self.arch_params.heads, "num_classes", new_num_classes)
            self.heads = factory.get(factory.insert_module_param(self.arch_params.heads, "in_channels", self.neck.out_channels))
            self._initialize_weights(self.arch_params)
