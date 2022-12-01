from super_gradients.modules import Residual, SkipConnection, BackboneInternalSkipConnection, HeadInternalSkipConnection, CrossModelSkipConnection

try:
    from pytorch_quantization import nn as quant_nn
    from super_gradients.training.utils.quantization.core import SGQuantMixin
    from super_gradients.training.utils.quantization.selective_quantization_utils import register_quantized_module

    _imported_pytorch_quantization_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    _imported_pytorch_quantization_failure = import_err


@register_quantized_module(float_source=Residual)
class QuantResidual(SGQuantMixin):
    """
    This is a placeholder module used by the quantization engine only.
    The module is to be used as a quantized substitute to a residual skip connection within a single block.
    """

    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure

    @classmethod
    def from_float(cls, float_instance: Residual, **kwargs):
        return quant_nn.TensorQuantizer(kwargs.get("quant_desc_input"))


@register_quantized_module(float_source=SkipConnection)
class QuantSkipConnection(SGQuantMixin):
    """
    This is a placeholder module used by the quantization engine only.
    The module is to be used as a quantized substitute to a skip connection between blocks.
    """

    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure

    @classmethod
    def from_float(cls, float_instance: SkipConnection, **kwargs):
        return quant_nn.TensorQuantizer(kwargs.get("quant_desc_input"))


@register_quantized_module(float_source=BackboneInternalSkipConnection)
class QuantBackboneInternalSkipConnection(QuantSkipConnection):
    """
    This is a placeholder module used by the quantization engine only.
    The module is to be used as a quantized substitute to a skip connection between blocks inside the backbone.
    """


@register_quantized_module(float_source=HeadInternalSkipConnection)
class QuantHeadInternalSkipConnection(QuantSkipConnection):
    """
    This is a placeholder module used by the quantization engine only.
    The module is to be used as a quantized substitute to a skip connection between blocks inside the head.
    """


@register_quantized_module(float_source=CrossModelSkipConnection)
class QuantCrossModelSkipConnection(QuantSkipConnection):
    """
    This is a placeholder module used by the quantization engine only.
    The module is to be used as a quantized substitute to a skip connection between backbone and the head.
    """
