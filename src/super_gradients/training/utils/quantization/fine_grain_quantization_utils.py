from typing import Tuple, Set, Type
from torch import nn

from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)
try:
    from pytorch_quantization.nn.modules._utils import QuantMixin, QuantInputMixin
    from pytorch_quantization.tensor_quant import QuantDescriptor
    from pytorch_quantization import nn as quant_nn

    from super_gradients.training.utils.quantization.core import SkipQuantization, SGQuantMixin, QuantizedMapping, \
        QuantizedMetadata

    _imported_pytorch_quantization_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.warning("Failed to import pytorch_quantization")
    _imported_pytorch_quantization_failure = import_err


class RegisterQuantizedModule(object):
    """
    Decorator used to register a Quantized module as a quantized version for Float module
    :param float_module:                the float module type that is being registered
    :param input_quant_descriptor:      the input quantization descriptor
    :param weights_quant_descriptor:    the weight quantization descriptor
    """
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure

    def __init__(self, *, float_module: Type[nn.Module], input_quant_descriptor=None, weights_quant_descriptor=None):
        self.float_module = float_module
        self.input_quant_descriptor = input_quant_descriptor
        self.weights_quant_descriptor = weights_quant_descriptor

    def __call__(self, quant_module: Type[SGQuantMixin]):
        QuantizationUtility.mapping_instructions.update({
            self.float_module: QuantizedMetadata(
                float_source=self.float_module,
                quantized_type=quant_module,
                input_quant_descriptor=self.input_quant_descriptor,
                weights_quant_descriptor=self.weights_quant_descriptor,
                action=QuantizedMetadata.ReplacementAction.REPLACE
            )
        })
        return quant_module  # this is required since the decorator assigns the result to the `quant_module`


class QuantizationUtility:

    """
    :param custom_mappings:                     custom mappings that extend the default mappings with extra behaviour
    :param default_quant_modules_calib_method:  default calibration method (default='percentile')
    :param default_per_channel_quant_modules:   whether quant modules should be per channel (default=False)
    """
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure
    mapping_instructions = {
        **{
            float_type: QuantizedMetadata(float_source=float_type, quantized_type=quantized_type,
                                          action=QuantizedMetadata.ReplacementAction.REPLACE)
            for (float_type, quantized_type) in [
                (nn.Conv2d, quant_nn.QuantConv2d),
                (nn.Linear, quant_nn.Linear),
                (nn.AvgPool2d, quant_nn.QuantAvgPool2d),
            ]
        },
        SkipQuantization: QuantizedMetadata(float_source=SkipQuantization, quantized_type=None,
                                            action=QuantizedMetadata.ReplacementAction.UNWRAP)
    }  # DEFAULT MAPPING INSTRUCTIONS

    def __init__(self, *, custom_mappings: dict = None, default_quant_modules_calib_method: str = 'percentile',
                 default_per_channel_quant_modules: bool = False) -> None:
        super().__init__()
        self.default_quant_modules_calib_method = default_quant_modules_calib_method
        self.default_per_channel_quant_modules = default_per_channel_quant_modules
        self.mapping_instructions = self.mapping_instructions.copy()
        if custom_mappings is not None:
            self.mapping_instructions.update(custom_mappings)  # OVERRIDE DEFAULT WITH CUSTOM. CUSTOM IS PRIORITIZED

    def _get_default_quant_descriptor(self):
        if self.default_quant_modules_calib_method in ["percentile", "mse", "entropy"]:
            calib_method_type = 'histogram'
        else:
            calib_method_type = 'max'

        if self.default_per_channel_quant_modules:
            return QuantDescriptor(calib_method=calib_method_type)

        return QuantDescriptor(calib_method=calib_method_type, axis=None)

    def register_skip_quantization(self, *, layer_names: Set[str]):
        self.mapping_instructions.update({
            name: QuantizedMetadata(float_source=name,
                                    quantized_type=None,
                                    action=QuantizedMetadata.ReplacementAction.SKIP)
            for name in layer_names
        })

    def register_quantization_mapping(self, *, layer_names: Set[str],
                                      quantized_type: Type[SGQuantMixin],
                                      input_quant_descriptor=None,
                                      weights_quant_descriptor=None):
        self.mapping_instructions.update({
            name: QuantizedMetadata(float_source=name,
                                    quantized_type=quantized_type,
                                    action=QuantizedMetadata.ReplacementAction.REPLACE,
                                    input_quant_descriptor=input_quant_descriptor,
                                    weights_quant_descriptor=weights_quant_descriptor)
            for name in layer_names
        })

    def _preprocess_skips_and_custom_mappings(self, module: nn.Module, nesting: Tuple[str, ...] = ()):
        """
        This pass is done to extract layer name and mapping instructions, so that we regard to per-layer processing.
        Relevant layer-specific mapping instructions are either `SkipQuantization` or `QuantizedMapping`, which are then
        being added to the mappings
        """
        mapping_instructions = dict()
        for name, child_module in module.named_children():
            nested_name = '.'.join(nesting + (name,))
            if isinstance(child_module, SkipQuantization):
                mapping_instructions[nested_name] = QuantizedMetadata(
                    float_source=nested_name,
                    quantized_type=None,
                    action=QuantizedMetadata.ReplacementAction.UNWRAP
                )

            if isinstance(child_module, QuantizedMapping):
                mapping_instructions[nested_name] = QuantizedMetadata(
                    float_source=nested_name,
                    quantized_type=child_module.quantized_type,
                    input_quant_descriptor=child_module.input_quant_descriptor,
                    weights_quant_descriptor=child_module.weights_quant_descriptor,
                    action=QuantizedMetadata.ReplacementAction.REPLACE
                )

            if isinstance(child_module, nn.Module):  # recursive call
                mapping_instructions.update(self._preprocess_skips_and_custom_mappings(child_module, nesting + (name,)))

        return mapping_instructions

    def _instantiate_quantized_from_float(self, float_module, metadata):
        base_classes = (QuantMixin, QuantInputMixin, SGQuantMixin)
        if not issubclass(metadata.quantized_type, base_classes):
            raise AssertionError(f"Quantization suite for {type(float_module).__name__} is invalid. "
                                 f"{metadata.quantized_type.__name__} must inherit one of "
                                 f"{', '.join(map(lambda _: _.__name__, base_classes))}")

        # USE PROVIDED QUANT DESCRIPTORS, OR DEFAULT IF NONE PROVIDED
        quant_descriptors = dict()
        if issubclass(metadata.quantized_type, (QuantMixin, QuantInputMixin)):
            quant_descriptors = {
                'quant_desc_input': metadata.input_quant_descriptor or self._get_default_quant_descriptor()
            }
        if issubclass(metadata.quantized_type, QuantMixin):
            quant_descriptors.update({
                'quant_desc_weight': (metadata.weights_quant_descriptor or self._get_default_quant_descriptor())
            })

        if not hasattr(metadata.quantized_type, 'from_float'):
            assert isinstance(metadata.quantized_type, SGQuantMixin), \
                f'{metadata.quantized_type.__name__} must inherit from ' \
                f'{SGQuantMixin.__name__}, so that it would include `from_float` class method'

        return metadata.quantized_type.from_float(float_module, **quant_descriptors)

    def _maybe_quantize_one_layer(self, module, child_name, nesting, child_module, mapping_instructions) -> bool:
        # if we don't have any instruction for the specific layer or the specific type - we continue
        # NOTE! IT IS IMPORTANT TO FIRST PROCESS THE NAME AND ONLY THEN THE TYPE
        if _imported_pytorch_quantization_failure is not None:
            raise _imported_pytorch_quantization_failure
        for candidate_key in ('.'.join(nesting + (child_name,)), type(child_module)):
            if candidate_key not in mapping_instructions:
                continue
            metadata: QuantizedMetadata = mapping_instructions[candidate_key]
            if metadata.action == QuantizedMetadata.ReplacementAction.SKIP:
                return True
            elif metadata.action == QuantizedMetadata.ReplacementAction.UNWRAP:
                assert isinstance(child_module, SkipQuantization)
                setattr(module, child_name, child_module.float_module)
                return True
            elif metadata.action == QuantizedMetadata.ReplacementAction.REPLACE:
                if isinstance(child_module, QuantizedMapping):  # UNWRAP MAPPING
                    child_module = child_module.float_module
                q_instance = self._instantiate_quantized_from_float(float_module=child_module, metadata=metadata)
                setattr(module, child_name, q_instance)
                return True
            else:
                raise NotImplementedError
        return False

    def quantize_module(self, module: nn.Module, nesting: Tuple[str, ...] = ()):
        per_layer_mappings = self._preprocess_skips_and_custom_mappings(module)
        mapping_instructions = {
            **per_layer_mappings,
            **self.mapping_instructions,
        }  # we first regard the per layer mappings, and then override with the custom mappings in case there is overlap

        for name, child_module in module.named_children():
            found = self._maybe_quantize_one_layer(module, name, nesting, child_module, mapping_instructions)

            # RECURSIVE CALL, to support module_list, sequential, custom (nested) modules
            if not found and isinstance(child_module, nn.Module):
                self.quantize_module(child_module)
