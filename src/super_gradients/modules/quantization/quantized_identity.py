from torch import nn

try:
    from pytorch_quantization import nn as quant_nn
    from super_gradients.training.utils.quantization.core import SGQuantMixin

    _imported_pytorch_quantization_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    _imported_pytorch_quantization_failure = import_err


class QuantIdentity(SGQuantMixin):
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure

    @classmethod
    def from_float(cls, float_instance: nn.Identity, **kwargs):
        return quant_nn.TensorQuantizer(kwargs.get("quant_desc_input"))
