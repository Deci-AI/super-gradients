from typing import Union

from torch import Tensor

from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.modules import RepVGGBlock
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.models.detection_models.csp_resnet import CSPResNet
from super_gradients.training.models.detection_models.pp_yolo_e.pan import CustomCSPPAN
from super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_head import PPYOLOEHead
from super_gradients.training.utils import HpmStruct
from super_gradients.training.models.arch_params_factory import get_arch_params


class PPYoloE(SgModule):
    def __init__(self, arch_params):
        super().__init__()
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()

        self.backbone = CSPResNet(**arch_params["backbone"], depth_mult=arch_params["depth_mult"], width_mult=arch_params["width_mult"])
        self.neck = CustomCSPPAN(**arch_params["neck"], depth_mult=arch_params["depth_mult"], width_mult=arch_params["width_mult"])
        self.head = PPYOLOEHead(**arch_params["head"], width_mult=arch_params["width_mult"], num_classes=arch_params["num_classes"])

    def forward(self, x: Tensor):
        features = self.backbone(x)
        features = self.neck(features)
        return self.head(features)

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        """
        Prepare the model to be converted to ONNX or other frameworks.
        Typically, this function will freeze the size of layers which is otherwise flexible, replace some modules
        with convertible substitutes and remove all auxiliary or training related parts.
        :param input_size: [H,W]
        """
        self.head.cache_anchors(input_size)

        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.fuse_block_residual_branches()

    def replace_head(self, new_num_classes=None, new_head=None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_head is not None:
            self.head = new_head
        else:
            self.head.replace_num_classes(new_num_classes)


@register_model(Models.PP_YOLOE_S)
class PPYoloE_S(PPYoloE):
    def __init__(self, arch_params):
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()
        arch_params = get_arch_params("ppyoloe_s_arch_params", overriding_params=arch_params)
        super().__init__(arch_params)


@register_model(Models.PP_YOLOE_M)
class PPYoloE_M(PPYoloE):
    def __init__(self, arch_params):
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()
        arch_params = get_arch_params("ppyoloe_m_arch_params", overriding_params=arch_params)
        super().__init__(arch_params)


@register_model(Models.PP_YOLOE_L)
class PPYoloE_L(PPYoloE):
    def __init__(self, arch_params):
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()
        arch_params = get_arch_params("ppyoloe_l_arch_params", overriding_params=arch_params)
        super().__init__(arch_params)


@register_model(Models.PP_YOLOE_X)
class PPYoloE_X(PPYoloE):
    def __init__(self, arch_params):
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()
        arch_params = get_arch_params("ppyoloe_x_arch_params", overriding_params=arch_params)
        super().__init__(arch_params)
