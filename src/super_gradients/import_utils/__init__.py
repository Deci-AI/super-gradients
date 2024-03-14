from .onnx_graphsurgeon import import_onnx_graphsurgeon_or_install, import_onnx_graphsurgeon_or_fail_with_instructions
from .pytorch_quantization import (
    import_pytorch_quantization_or_install,
    import_pytorch_quantization_or_fail_with_instructions,
    patch_pytorch_quantization_modules_if_needed,
)

__all__ = [
    "import_onnx_graphsurgeon_or_install",
    "import_onnx_graphsurgeon_or_fail_with_instructions",
    "import_pytorch_quantization_or_install",
    "import_pytorch_quantization_or_fail_with_instructions",
    "patch_pytorch_quantization_modules_if_needed",
]
