from pytorch_quantization import nn as quant_nn

from super_gradients.training.models.classification_models.resnet import Bottleneck
from super_gradients.training.utils.quantization.core import SGQuantMixin, QuantizedMetadata
from super_gradients.training.utils.quantization.selective_quantization_utils import register_quantized_module


@register_quantized_module(float_source=Bottleneck, action=QuantizedMetadata.ReplacementAction.RECURE_AND_REPLACE)
class QuantBottleneck(SGQuantMixin):
    """
    we just insert quantized tensor to the shortcut (=residual) layer, so that it would be quantized
    NOTE: we must quantize the float instance, so the mode should be
          QuantizedMetadata.ReplacementAction.RECURE_AND_REPLACE
    """

    @classmethod
    def from_float(cls, float_instance: Bottleneck, **kwargs):
        float_instance.shortcut.add_module("residual_quantizer", quant_nn.TensorQuantizer(kwargs.get("quant_desc_input")))
        return float_instance
