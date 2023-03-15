import copy

from super_gradients.common.environment.cfg_utils import load_arch_params
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.utils import HpmStruct
import hydra

arch_params = cfg = hydra.utils.instantiate(load_arch_params("deciyolo_m_arch_params"))
model = CustomizableDetector(HpmStruct(**copy.deepcopy(arch_params)))
