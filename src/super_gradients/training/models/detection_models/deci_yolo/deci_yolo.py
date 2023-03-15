from super_gradients.common.environment.cfg_utils import load_arch_params
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector

arch_params = load_arch_params("deciyolo_m_arch_params")
model = CustomizableDetector(arch_params)
