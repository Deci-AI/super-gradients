from typing import Mapping, Any, Union

from super_gradients.modules import RepVGGBlock
from super_gradients.training.models.detection_models.csp_resnet import CSPResNet
from super_gradients.training.models.detection_models.pp_yolo_e.pan import CustomCSPPAN
from super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_head import PPYOLOEHead
from torch import nn, Tensor


class PPYoloE(nn.Module):
    def __init__(self, **arch_params):
        super().__init__()
        self.backbone = CSPResNet(**arch_params["backbone"])
        self.neck = CustomCSPPAN(**arch_params["neck"])
        self.head = PPYOLOEHead(**arch_params["head"])

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
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.prep_model_for_conversion(input_size)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]):
        return cls(
            depth_mult=config.depth_mult,
            width_mult=config.width_mult,
        )

    @classmethod
    def from_pretrained(cls, variant: str):

        variants = {"ppyolo_s": {}}

        return cls.from_config(variants[variant])
