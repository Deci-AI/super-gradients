from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)
try:
    from super_gradients.training.utils.quantization.core import _inject_class_methods_to_default_quant_types

    _inject_class_methods_to_default_quant_types()
except (ImportError, NameError, ModuleNotFoundError):
    logger.warning("Failed to import pytorch_quantization")
