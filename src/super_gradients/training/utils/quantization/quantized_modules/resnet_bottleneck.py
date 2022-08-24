from typing import Union, Type

from pytorch_quantization.tensor_quant import QuantDescriptor

from super_gradients.training.models import Bottleneck
from super_gradients.training.utils.quantization.core import SGQuantMixin, QuantizedMetadata
from pytorch_quantization import nn as quant_nn


class QuantBottleneck(SGQuantMixin):
    """
    we just insert quantized tensor to the shortcut (=residual) layer, so that it would be quantized
    NOTE: we must quantize the float instance, so the mode should be
          QuantizedMetadata.ReplacementAction.QUANTIZE_CHILDREN_THEN_REPLACE
    """

    @classmethod
    def from_float(cls, float_instance: Bottleneck, **kwargs):
        float_instance.shortcut.add_module("residual_quantizer",
                                           quant_nn.TensorQuantizer(kwargs.get('quant_desc_input')))
        return float_instance

    @staticmethod
    def get_quantized_metadata_for_source(float_source: Union[str, Type]):
        return QuantizedMetadata(float_source=float_source,
                                 quantized_target_class=QuantBottleneck,
                                 input_quant_descriptor=QuantDescriptor(calib_method='histogram'),
                                 action=QuantizedMetadata.ReplacementAction.QUANTIZE_CHILD_MODULES_THEN_REPLACE)
