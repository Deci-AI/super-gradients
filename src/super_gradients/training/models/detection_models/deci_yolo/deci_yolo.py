import copy

import torch
from super_gradients.common.environment.cfg_utils import load_arch_params
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.utils import HpmStruct
import hydra


if __name__ == "__main__":
    arch_params = hydra.utils.instantiate(load_arch_params("deciyolo_s_arch_params"))
    model = CustomizableDetector(HpmStruct(**copy.deepcopy(arch_params))).cuda()

    input = torch.randn((1, 3, 416, 416)).cuda()

    model.train().forward(input)
    model.eval().forward(input)
