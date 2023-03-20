"""
A base for a detection network built according to the following scheme:
 * constructed from nested arch_params;
 * inside arch_params each nested level (module) has an explicit type and its required parameters
 * each module accepts in_channels and other parameters
 * each module defines out_channels property on construction
"""


from typing import Union, Optional

from torch import nn


from super_gradients.training.models.sg_module import SgModule
import super_gradients.common.factories.detection_modules_factory as det_factory


class CustomizableDetector(SgModule):
    """
    A customizable detector with backbone -> neck -> heads
    Each submodule with its parameters must be defined explicitly.
    Modules should follow the interface of BaseDetectionModule
    """

    def __init__(
        self,
        backbone: dict,
        heads: dict,
        neck: dict = None,
        num_classes: int = None,
        bn_eps: Optional[float] = None,
        bn_momentum: Optional[float] = None,
        inplace_act: Optional[bool] = True,
        in_channels: int = 3,
    ):
        """
        :param type_mapping: can be passed to resolve string type names in arch_params to actual types
        """
        super().__init__()

        self.heads_params = heads
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.inplace_act = inplace_act
        factory = det_factory.DetectionModulesFactory()

        # move num_classes into heads params
        if num_classes:
            self.heads_params = factory.insert_module_param(self.heads_params, "num_classes", num_classes)

        self.backbone = factory.get(factory.insert_module_param(backbone, "in_channels", in_channels))
        if neck is not None:
            self.neck = factory.get(factory.insert_module_param(neck, "in_channels", self.backbone.out_channels))
            self.heads = factory.get(factory.insert_module_param(heads, "in_channels", self.neck.out_channels))
        else:
            self.neck = nn.Identity()
            self.heads = factory.get(factory.insert_module_param(heads, "in_channels", self.backbone.out_channels))

        self._initialize_weights(self.bn_eps, self.bn_momentum, self.inplace_act)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.heads(x)

    def _initialize_weights(self, bn_eps: float, bn_momentum: float, inplace_act: bool):
        for m in self.modules():
            t = type(m)
            if t is nn.BatchNorm2d:
                m.eps = bn_eps if bn_eps else m.eps
                m.momentum = bn_momentum if bn_momentum else m.momentum
            elif inplace_act and t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, nn.Mish]:
                m.inplace = True

    def prep_model_for_conversion(self, input_size: Optional[Union[tuple, list]] = None, **kwargs):
        for module in self.modules():
            if module != self and hasattr(module, "prep_model_for_conversion"):
                module.prep_model_for_conversion(input_size, **kwargs)

    def replace_head(self, new_num_classes: Optional[int] = None, new_head: Optional[nn.Module] = None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_head is not None:
            self.heads = new_head
        else:
            factory = det_factory.DetectionModulesFactory()
            self.heads_params = factory.insert_module_param(self.heads_params, "num_classes", new_num_classes)
            self.heads = factory.get(factory.insert_module_param(self.heads_params, "in_channels", self.neck.out_channels))
            self._initialize_weights(self.bn_eps, self.bn_momentum, self.inplace_act)
