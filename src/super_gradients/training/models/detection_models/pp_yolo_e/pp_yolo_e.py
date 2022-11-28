from typing import Union

from super_gradients.modules import RepVGGBlock
from super_gradients.modules.normalize_input import NormalizeInput
from super_gradients.training.models import SgModule
from super_gradients.training.models.detection_models.csp_resnet import CSPResNet
from super_gradients.training.models.detection_models.pp_yolo_e.pan import CustomCSPPAN
from super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_head import PPYOLOEHead
from torch import Tensor, nn


class PPYoloE(SgModule):
    def __init__(self, arch_params):
        super().__init__()
        arch_params = arch_params.to_dict()

        if "normalization" in arch_params and arch_params["normalization"] is not None:
            self.normalize = NormalizeInput(**arch_params["normalization"])
        else:
            self.normalize = nn.Identity()
        self.backbone = CSPResNet(**arch_params["backbone"])
        self.neck = CustomCSPPAN(**arch_params["neck"])
        self.head = PPYOLOEHead(**arch_params["head"])

    def forward(self, x: Tensor):
        x = self.normalize(x)
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
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.prep_model_for_conversion(input_size)
