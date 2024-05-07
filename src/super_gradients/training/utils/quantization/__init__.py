from super_gradients.import_utils import import_pytorch_quantization_or_fail_with_instructions

import_pytorch_quantization_or_fail_with_instructions()

from .calibrator import QuantizationCalibrator  # noqa: E402
from .core import (  # noqa: E402
    _inject_class_methods_to_default_quant_types,
    SGQuantMixin,
    SkipQuantization,
    QuantizedMetadata,
    QuantizedMapping,
)
from .export import export_quantized_module_to_onnx  # noqa: E402
from .selective_quantization_utils import SelectiveQuantizer, register_quantized_module  # noqa: E402
from .modules import (  # noqa: E402
    QuantSTDCBlock,
    QuantAttentionRefinementModule,
    QuantFeatureFusionModule,
    QuantContextPath,
    QuantBottleneck,
    QuantResidual,
    QuantCrossModelSkipConnection,
    QuantBackboneInternalSkipConnection,
    QuantHeadInternalSkipConnection,
    QuantSkipConnection,
)  # noqa: E402
from .use_fb_fake_quant import use_fb_fake_quant  # noqa: E402
from .ptq import ptq  # noqa: E402

_inject_class_methods_to_default_quant_types()

__all__ = [
    "SelectiveQuantizer",
    "register_quantized_module",
    "SGQuantMixin",
    "SkipQuantization",
    "QuantizedMetadata",
    "QuantizedMapping",
    "QuantizationCalibrator",
    "QuantSTDCBlock",
    "QuantAttentionRefinementModule",
    "QuantFeatureFusionModule",
    "QuantContextPath",
    "QuantBottleneck",
    "QuantResidual",
    "QuantCrossModelSkipConnection",
    "QuantBackboneInternalSkipConnection",
    "QuantHeadInternalSkipConnection",
    "QuantSkipConnection",
    "export_quantized_module_to_onnx",
    "use_fb_fake_quant",
    "ptq",
]
