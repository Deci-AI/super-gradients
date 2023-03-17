import copy
from typing import Union

import torch
from omegaconf import DictConfig
from super_gradients.common.environment.cfg_utils import load_arch_params
from super_gradients.common.object_names import Models
from super_gradients.common.registry import register_model
from super_gradients.training.models.arch_params_factory import get_arch_params
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.utils import HpmStruct
import hydra


@register_model(Models.DECIYOLO_S)
class DeciYolo_S(CustomizableDetector):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig], in_channels: int = 3):
        default_arch_params = get_arch_params("deciyolo_s_arch_params")
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(merged_arch_params, in_channels=in_channels)

    # def __init__(self, num_classes: int):
    #     arch_params = load_arch_params("deciyolo_s_arch_params")
    #     arch_params = hydra.utils.instantiate(arch_params)
    #     arch_params["heads"][list(arch_params["heads"].keys())[0]]["num_classes"] = num_classes
    #     super(DeciYolo_S, self).__init__(HpmStruct(**copy.deepcopy(arch_params)))

    @property
    def num_classes(self):
        return self.heads.num_classes


@register_model(Models.DECIYOLO_M)
class DeciYolo_M(CustomizableDetector):
    def __init__(self, num_classes: int):
        arch_params = load_arch_params("deciyolo_m_arch_params")
        arch_params = hydra.utils.instantiate(arch_params)
        arch_params["heads"][list(arch_params["heads"].keys())[0]]["num_classes"] = num_classes
        super(DeciYolo_M, self).__init__(HpmStruct(**copy.deepcopy(arch_params)))

    @property
    def num_classes(self):
        return self.heads.num_classes


@register_model(Models.DECIYOLO_L)
class DeciYolo_L(CustomizableDetector):
    def __init__(self, num_classes: int):
        arch_params = load_arch_params("deciyolo_l_arch_params")
        arch_params = hydra.utils.instantiate(arch_params)
        arch_params["heads"][list(arch_params["heads"].keys())[0]]["num_classes"] = num_classes
        super(DeciYolo_L, self).__init__(HpmStruct(**copy.deepcopy(arch_params)))

    @property
    def num_classes(self):
        return self.heads.num_classes


if __name__ == "__main__":
    model = DeciYolo_S(num_classes=80)
    # arch_params = hydra.utils.instantiate(load_arch_params("deciyolo_m_arch_params"))
    # model = CustomizableDetector(HpmStruct(**copy.deepcopy(arch_params))).cuda()

    input = torch.randn((1, 3, 416, 416)).cuda()

    model.train().forward(input)
    model.eval().forward(input)
