from super_gradients.training.models import Bottleneck

try:
    from pytorch_quantization import nn as quant_nn
    from super_gradients.training.utils.quantization.core import SGQuantMixin

    _imported_pytorch_quantization_failure = False

except (ImportError, NameError, ModuleNotFoundError):
    _imported_pytorch_quantization_failure = True


class QuantBottleneck(SGQuantMixin):
    """
    we just insert quantized tensor to the shortcut (=residual) layer, so that it would be quantized
    NOTE: we must quantize the float instance, so the mode should be
          QuantizedMetadata.ReplacementAction.QUANTIZE_CHILDREN_THEN_REPLACE
    """

    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure

    @classmethod
    def from_float(cls, float_instance: Bottleneck, **kwargs):
        float_instance.shortcut.add_module("residual_quantizer", quant_nn.TensorQuantizer(kwargs.get("quant_desc_input")))
        return float_instance
