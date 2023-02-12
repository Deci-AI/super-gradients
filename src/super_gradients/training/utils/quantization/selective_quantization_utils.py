import logging
from typing import Tuple, Set, Type, Dict, Union, Callable, Optional
from torch import nn

from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)
try:
    from pytorch_quantization.nn.modules._utils import QuantMixin, QuantInputMixin
    from pytorch_quantization.tensor_quant import QuantDescriptor
    from pytorch_quantization import nn as quant_nn

    from super_gradients.training.utils.quantization.core import SkipQuantization, SGQuantMixin, QuantizedMapping, QuantizedMetadata

    _imported_pytorch_quantization_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.warning("Failed to import pytorch_quantization")
    _imported_pytorch_quantization_failure = import_err


def register_quantized_module(
    float_source: Union[str, Type[nn.Module]],
    action: QuantizedMetadata.ReplacementAction = QuantizedMetadata.ReplacementAction.REPLACE,
    input_quant_descriptor: Optional[QuantDescriptor] = None,
    weights_quant_descriptor: Optional[QuantDescriptor] = None,
) -> Callable:
    """
    Decorator used to register a Quantized module as a quantized version for Float module
    :param action:                      action to perform on the float_source
    :param float_source:                the float module type that is being registered
    :param input_quant_descriptor:      the input quantization descriptor
    :param weights_quant_descriptor:    the weight quantization descriptor
    """

    def decorator(quant_module: Type[SGQuantMixin]) -> Type[SGQuantMixin]:
        if float_source in SelectiveQuantizer.mapping_instructions:
            metadata = SelectiveQuantizer.mapping_instructions[float_source]
            raise ValueError(f"`{float_source}` is already registered with following metadata {metadata}")

        SelectiveQuantizer.mapping_instructions.update(
            {
                float_source: QuantizedMetadata(
                    float_source=float_source,
                    quantized_target_class=quant_module,
                    input_quant_descriptor=input_quant_descriptor,
                    weights_quant_descriptor=weights_quant_descriptor,
                    action=action,
                )
            }
        )
        return quant_module  # this is required since the decorator assigns the result to the `quant_module`

    return decorator


class SelectiveQuantizer:
    """
    :param custom_mappings:                             custom mappings that extend the default mappings with extra behaviour
    :param default_per_channel_quant_weights:           whether quant module weights should be per channel (default=True)
    :param default_quant_modules_calibrator_weights:    default calibrator method for weights (default='max')
    :param default_quant_modules_calibrator_inputs:     default calibrator method for inputs (default='histogram')
    :param default_learn_amax:                          EXPERIMENTAL! whether quant modules should have learnable amax (default=False)
    """

    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure

    mapping_instructions: Dict[Union[str, Type], QuantizedMetadata] = {
        **{
            float_type: QuantizedMetadata(
                float_source=float_type,
                quantized_target_class=quantized_target_class,
                action=QuantizedMetadata.ReplacementAction.REPLACE,
            )
            for (float_type, quantized_target_class) in [
                (nn.Conv1d, quant_nn.QuantConv1d),
                (nn.Conv2d, quant_nn.QuantConv2d),
                (nn.Conv3d, quant_nn.QuantConv3d),
                (nn.ConvTranspose1d, quant_nn.QuantConvTranspose1d),
                (nn.ConvTranspose2d, quant_nn.QuantConvTranspose2d),
                (nn.ConvTranspose3d, quant_nn.QuantConvTranspose3d),
                (nn.Linear, quant_nn.Linear),
                (nn.LSTM, quant_nn.LSTM),
                (nn.LSTMCell, quant_nn.LSTMCell),
                (nn.AvgPool1d, quant_nn.QuantAvgPool1d),
                (nn.AvgPool2d, quant_nn.QuantAvgPool2d),
                (nn.AvgPool3d, quant_nn.QuantAvgPool3d),
                (nn.AdaptiveAvgPool1d, quant_nn.QuantAdaptiveAvgPool1d),
                (nn.AdaptiveAvgPool2d, quant_nn.QuantAdaptiveAvgPool2d),
                (nn.AdaptiveAvgPool3d, quant_nn.QuantAdaptiveAvgPool3d),
            ]
        },
        SkipQuantization: QuantizedMetadata(float_source=SkipQuantization, quantized_target_class=None, action=QuantizedMetadata.ReplacementAction.UNWRAP),
    }  # DEFAULT MAPPING INSTRUCTIONS

    def __init__(
        self,
        *,
        custom_mappings: dict = None,
        default_quant_modules_calibrator_weights: str = "max",
        default_quant_modules_calibrator_inputs: str = "histogram",
        default_per_channel_quant_weights: bool = True,
        default_learn_amax: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.default_quant_modules_calibrator_weights = default_quant_modules_calibrator_weights
        self.default_quant_modules_calibrator_inputs = default_quant_modules_calibrator_inputs
        self.default_per_channel_quant_weights = default_per_channel_quant_weights
        self.default_learn_amax = default_learn_amax
        self.verbose = verbose
        self.mapping_instructions = self.mapping_instructions.copy()
        if custom_mappings is not None:
            self.mapping_instructions.update(custom_mappings)  # OVERRIDE DEFAULT WITH CUSTOM. CUSTOM IS PRIORITIZED

    def _get_default_quant_descriptor(self, for_weights=False):
        methods = {"percentile": "histogram", "mse": "histogram", "entropy": "histogram", "histogram": "histogram", "max": "max"}

        if for_weights:
            axis = 0 if self.default_per_channel_quant_weights else None

            learn_amax = self.default_learn_amax
            if self.default_learn_amax and self.default_per_channel_quant_weights:
                logger.error("Learnable amax is suported only for per-tensor quantization. Disabling it for weights quantization!")
                learn_amax = False

            return QuantDescriptor(calib_method=methods[self.default_quant_modules_calibrator_weights], axis=axis, learn_amax=learn_amax)
        else:
            # activations stay per-tensor by default
            return QuantDescriptor(calib_method=methods[self.default_quant_modules_calibrator_inputs], learn_amax=self.default_learn_amax)

    def register_skip_quantization(self, *, layer_names: Optional[Set[str]] = None):
        if layer_names is not None:
            self.mapping_instructions.update(
                {
                    name: QuantizedMetadata(float_source=name, quantized_target_class=None, action=QuantizedMetadata.ReplacementAction.SKIP)
                    for name in layer_names
                }
            )

    def register_quantization_mapping(
        self, *, layer_names: Set[str], quantized_target_class: Type[SGQuantMixin], input_quant_descriptor=None, weights_quant_descriptor=None
    ):
        self.mapping_instructions.update(
            {
                name: QuantizedMetadata(
                    float_source=name,
                    quantized_target_class=quantized_target_class,
                    action=QuantizedMetadata.ReplacementAction.REPLACE,
                    input_quant_descriptor=input_quant_descriptor,
                    weights_quant_descriptor=weights_quant_descriptor,
                )
                for name in layer_names
            }
        )

    def _preprocess_skips_and_custom_mappings(self, module: nn.Module, nesting: Tuple[str, ...] = ()):
        """
        This pass is done to extract layer name and mapping instructions, so that we regard to per-layer processing.
        Relevant layer-specific mapping instructions are either `SkipQuantization` or `QuantizedMapping`, which are then
        being added to the mappings
        """
        mapping_instructions = dict()
        for name, child_module in module.named_children():
            nested_name = ".".join(nesting + (name,))
            if isinstance(child_module, SkipQuantization):
                mapping_instructions[nested_name] = QuantizedMetadata(
                    float_source=nested_name, quantized_target_class=None, action=QuantizedMetadata.ReplacementAction.UNWRAP
                )

            if isinstance(child_module, QuantizedMapping):
                mapping_instructions[nested_name] = QuantizedMetadata(
                    float_source=nested_name,
                    quantized_target_class=child_module.quantized_target_class,
                    input_quant_descriptor=child_module.input_quant_descriptor,
                    weights_quant_descriptor=child_module.weights_quant_descriptor,
                    action=child_module.action,
                )

            if isinstance(child_module, nn.Module):  # recursive call
                mapping_instructions.update(self._preprocess_skips_and_custom_mappings(child_module, nesting + (name,)))

        return mapping_instructions

    def _instantiate_quantized_from_float(self, float_module, metadata, preserve_state_dict):
        base_classes = (QuantMixin, QuantInputMixin, SGQuantMixin)
        if not issubclass(metadata.quantized_target_class, base_classes):
            raise AssertionError(
                f"Quantization suite for {type(float_module).__name__} is invalid. "
                f"{metadata.quantized_target_class.__name__} must inherit one of "
                f"{', '.join(map(lambda _: _.__name__, base_classes))}"
            )

        # USE PROVIDED QUANT DESCRIPTORS, OR DEFAULT IF NONE PROVIDED
        quant_descriptors = dict()
        if issubclass(metadata.quantized_target_class, (SGQuantMixin, QuantMixin, QuantInputMixin)):
            quant_descriptors = {"quant_desc_input": metadata.input_quant_descriptor or self._get_default_quant_descriptor(for_weights=False)}
        if issubclass(metadata.quantized_target_class, (SGQuantMixin, QuantMixin)):
            quant_descriptors.update({"quant_desc_weight": metadata.weights_quant_descriptor or self._get_default_quant_descriptor(for_weights=True)})

        if not hasattr(metadata.quantized_target_class, "from_float"):
            assert isinstance(metadata.quantized_target_class, SGQuantMixin), (
                f"{metadata.quantized_target_class.__name__} must inherit from " f"{SGQuantMixin.__name__}, so that it would include `from_float` class method"
            )

        q_instance = metadata.quantized_target_class.from_float(float_module, **quant_descriptors)

        # MOVE TENSORS TO ORIGINAL DEVICE
        if len(list(float_module.parameters(recurse=False))) > 0:
            q_instance = q_instance.to(next(float_module.parameters(recurse=False)).device)
        elif len(list(float_module.buffers(recurse=False))):
            q_instance = q_instance.to(next(float_module.buffers(recurse=False)).device)

        # COPY STATE DICT IF NEEDED
        if preserve_state_dict:
            # quant state dict may have additional parameters for Clip and strict loading will fail
            # if we find at least one Clip module in q_instance, disable strict loading and hope for the best
            strict_load = True
            for k in q_instance.state_dict().keys():
                if "clip.clip_value_max" in k or "clip.clip_value_min" in k:
                    strict_load = False
                    logger.debug(
                        "Instantiating quant module in non-strict mode leaving Clip parameters non-initilaized. Use QuantizationCalibrator to initialize them."
                    )
                    break

            q_instance.load_state_dict(float_module.state_dict(), strict=strict_load)

        return q_instance

    def _maybe_quantize_one_layer(
        self,
        module: nn.Module,
        child_name: str,
        nesting: Tuple[str, ...],
        child_module: nn.Module,
        mapping_instructions: Dict[Union[str, Type], QuantizedMetadata],
        preserve_state_dict: bool,
    ) -> bool:
        """
        Does the heavy lifting of (maybe) quantizing a layer: creates a quantized instance based on a float instance,
        and replaces it in the "parent" module

        :param module:                  the module we'd like to quantize a specific layer in
        :param child_name:              the attribute name of the layer in the module
        :param nesting:                 the current nesting we're in. Needed to find the appropriate key in the mappings
        :param child_module:            the instance of the float module we'd like to quantize
        :param mapping_instructions:    mapping instructions: how to quantize
        :param preserve_state_dict:     whether to copy the state dict from the float instance to the quantized instance

        :return: a boolean indicates if we found a match and should not continue recursively
        """
        # if we don't have any instruction for the specific layer or the specific type - we continue
        # NOTE! IT IS IMPORTANT TO FIRST PROCESS THE NAME AND ONLY THEN THE TYPE
        if _imported_pytorch_quantization_failure is not None:
            raise _imported_pytorch_quantization_failure
        for candidate_key in (".".join(nesting + (child_name,)), type(child_module)):
            if candidate_key not in mapping_instructions:
                continue
            metadata: QuantizedMetadata = mapping_instructions[candidate_key]
            if metadata.action == QuantizedMetadata.ReplacementAction.SKIP:
                return True
            elif metadata.action == QuantizedMetadata.ReplacementAction.UNWRAP:
                assert isinstance(child_module, SkipQuantization)
                setattr(module, child_name, child_module.float_module)
                return True
            elif metadata.action in (
                QuantizedMetadata.ReplacementAction.REPLACE,
                QuantizedMetadata.ReplacementAction.REPLACE_AND_RECURE,
                QuantizedMetadata.ReplacementAction.RECURE_AND_REPLACE,
            ):
                if isinstance(child_module, QuantizedMapping):  # UNWRAP MAPPING
                    child_module = child_module.float_module
                q_instance: nn.Module = self._instantiate_quantized_from_float(
                    float_module=child_module, metadata=metadata, preserve_state_dict=preserve_state_dict
                )

                # ACTUAL REPLACEMENT
                def replace():
                    setattr(module, child_name, q_instance)

                def recurse_quantize():
                    self._quantize_module_aux(
                        module=getattr(module, child_name),
                        mapping_instructions=mapping_instructions,
                        nesting=nesting + (child_name,),
                        preserve_state_dict=preserve_state_dict,
                    )

                if metadata.action == QuantizedMetadata.ReplacementAction.REPLACE:
                    replace()

                elif metadata.action == QuantizedMetadata.ReplacementAction.REPLACE_AND_RECURE:
                    replace()
                    recurse_quantize()
                elif metadata.action == QuantizedMetadata.ReplacementAction.RECURE_AND_REPLACE:
                    recurse_quantize()
                    replace()
                return True
            else:
                raise NotImplementedError
        return False

    def quantize_module(self, module: nn.Module, *, preserve_state_dict=True):
        per_layer_mappings = self._preprocess_skips_and_custom_mappings(module)
        mapping_instructions = {
            **per_layer_mappings,
            **self.mapping_instructions,
        }  # we first regard the per layer mappings, and then override with the custom mappings in case there is overlap
        logging_level = logging.getLogger("absl").getEffectiveLevel()
        if not self.verbose:  # suppress pytorch-quantization spam
            logging.getLogger("absl").setLevel("ERROR")

        device = next(module.parameters()).device
        self._quantize_module_aux(mapping_instructions=mapping_instructions, module=module, nesting=(), preserve_state_dict=preserve_state_dict)
        module.to(device)

        logging.getLogger("absl").setLevel(logging_level)

    def _quantize_module_aux(self, mapping_instructions, module, nesting, preserve_state_dict):
        for name, child_module in module.named_children():
            found = self._maybe_quantize_one_layer(module, name, nesting, child_module, mapping_instructions, preserve_state_dict)

            # RECURSIVE CALL, to support module_list, sequential, custom (nested) modules
            if not found and isinstance(child_module, nn.Module):
                self._quantize_module_aux(mapping_instructions, child_module, nesting + (name,), preserve_state_dict)
