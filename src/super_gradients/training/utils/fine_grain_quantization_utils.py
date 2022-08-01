from typing import Tuple, Set, Union
from dataclasses import dataclass
from pytorch_quantization.nn.modules._utils import QuantMixin, QuantInputMixin
from pytorch_quantization.tensor_quant import QuantDescriptor
from torch import nn
from pytorch_quantization import nn as quant_nn

from super_gradients.training.utils.quantization.core import SkipQuantization, SGQuantMixin


@dataclass(init=True)
class QuantizedMetadata:
    quantized_type: Union[QuantMixin, QuantInputMixin]
    input_quant_descriptor: QuantDescriptor = None  # default is used if None
    weights_quant_descriptor: QuantDescriptor = None  # default is used if None


class RegisterQuantizedModule(object):
    def __init__(self, *, float_module, input_quant_descriptor=None, weights_quant_descriptor=None):
        self.float_module = float_module
        self.input_quant_descriptor = input_quant_descriptor
        self.weights_quant_descriptor = weights_quant_descriptor

    def __call__(self, quant_module):

        QuantizationUtility.mappings.update({
            self.float_module: QuantizedMetadata(
                quantized_type=quant_module,
                input_quant_descriptor=self.input_quant_descriptor,
                weights_quant_descriptor=self.weights_quant_descriptor
            )
        })


class QuantizationUtility:

    mappings = {
        SkipQuantization: None,  # SKIP
        nn.Conv2d: QuantizedMetadata(quantized_type=quant_nn.QuantConv2d),
        nn.Linear: QuantizedMetadata(quantized_type=quant_nn.Linear),
        nn.AvgPool2d: QuantizedMetadata(quantized_type=quant_nn.QuantAvgPool2d),
    }

    def __init__(self, *, custom_mappings: dict = None, default_quant_modules_calib_method: str = 'percentile',
                 default_per_channel_quant_modules: bool = False) -> None:
        super().__init__()
        self.default_quant_modules_calib_method = default_quant_modules_calib_method
        self.default_per_channel_quant_modules = default_per_channel_quant_modules
        self.mappings = self.mappings.copy()
        if custom_mappings is not None:
            self.mappings.update(custom_mappings)  # OVERRIDE DEFAULT WITH CUSTOM. CUSTOM IS PRIORITIZED

    def _get_default_quant_descriptor(self):
        if self.default_quant_modules_calib_method in ["percentile", "mse", "entropy"]:
            calib_method_type = 'histogram'
        else:
            calib_method_type = 'max'

        if self.default_per_channel_quant_modules:
            return QuantDescriptor(calib_method=calib_method_type)

        return QuantDescriptor(calib_method=calib_method_type, axis=None)

    def wrap_with_skip_quantization(self, module: nn.Module, layer_names: Set[str], nesting: Tuple[str, ...] = ()):
        for name, child_module in module.named_children():
            nested_name = '.'.join(nesting + (name,))
            if nested_name in layer_names:
                layer_names.remove(nested_name)
                setattr(module, name, SkipQuantization(child_module))

            # RECURSIVE CALL, to support module_list, sequential, custom (nested) modules
            if isinstance(child_module, nn.Module):
                self.wrap_with_skip_quantization(child_module, layer_names, nesting + (name,))

    def quantize_module(self, module: nn.Module):
        base_classes = (QuantMixin, QuantInputMixin, SGQuantMixin)
        for name, child_module in module.named_children():
            if type(child_module) in self.mappings:
                quant_suite: QuantizedMetadata = self.mappings[type(child_module)]
                if quant_suite is None:  # SKIP QUANTIZATION
                    continue
                if quant_suite.quantized_type is None:
                    raise AssertionError(f"Quantization suite for {type(child_module).__name__} is incomplete. "
                                         f"Please add `quantized_type`")

                if not issubclass(quant_suite.quantized_type, base_classes):
                    raise AssertionError(f"Quantization suite for {type(child_module).__name__} is invalid. "
                                         f"{quant_suite.quantized_type.__name__} must inherit one of "
                                         f"{', '.join(map(lambda _: _.__name__, base_classes))}")

                # USE PROVIDED QUANT DESCRIPTORS, OR DEFAULT IF NONE PROVIDED
                quant_descriptors = dict()
                if issubclass(quant_suite.quantized_type, (QuantMixin, QuantInputMixin)):
                    quant_descriptors = {
                        'quant_desc_input': quant_suite.input_quant_descriptor or self._get_default_quant_descriptor()
                    }
                if issubclass(quant_suite.quantized_type, QuantMixin):
                    quant_descriptors.update({
                        'quant_desc_weight': (quant_suite.weights_quant_descriptor
                                              or self._get_default_quant_descriptor())
                    })

                if not hasattr(quant_suite.quantized_type, 'from_float'):
                    assert isinstance(quant_suite.quantized_type, SGQuantMixin), \
                        f'{quant_suite.quantized_type.__name__} must inherit from ' \
                        f'{SGQuantMixin.__name__}, so that it would include `from_float` class method'

                # ACTUAL REPLACEMENT
                quant_child_module = quant_suite.quantized_type.from_float(child_module, **quant_descriptors)
                setattr(module, name, quant_child_module)

            # RECURSIVE CALL, to support module_list, sequential, custom (nested) modules
            if isinstance(child_module, nn.Module):
                self.quantize_module(child_module)
