import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Union, Type, Optional, Set

from pytorch_quantization.nn.modules._utils import QuantMixin, QuantInputMixin
from pytorch_quantization.tensor_quant import QuantDescriptor
from torch import nn


def _extract_init_args(cls, float_instance, ignore_init_args: Set[str] = ()):
    """
    Inspecting the __init__ args, and searching for corresponding properties from the float instance
    e.g., for `__init__(self, a)` the mechanism will look for `float_instance.a` and pass that value to `__init__`
    """
    required_init_params = list(inspect.signature(cls.__init__).parameters)[1:]  # [0] is self

    if "kwargs" in required_init_params:  # we don't want to search for a state named `kwargs`
        required_init_params.pop(required_init_params.index("kwargs"))

    float_instance_state = {}
    for p in required_init_params:
        if p in ignore_init_args:  # ignore these args and don't pick state from the instance
            continue
        if not hasattr(float_instance, p):
            raise ValueError(
                f"{float_instance.__class__.__name__} is missing `{p}` which is required "
                f"in {cls.__name__}.__init__. Either override `SGQuantBase.from_float` "
                f"or add {p} as state for {float_instance.__class__.__name__}."
            )
        float_instance_state[p] = getattr(float_instance, p)

    # Edge-cases
    if "bias" in float_instance_state:
        if float_instance_state["bias"] is None:  # None is the state when bias=False in torch.nn
            float_instance_state["bias"] = False
        elif not isinstance(float_instance_state["bias"], bool):  # Tensor is the state when bias=True in torch.nn
            float_instance_state["bias"] = True
        # in case bias is a boolean - we don't do anything, so it is taken as-is, either True or False
    return float_instance_state


def _from_float(cls, float_instance, ignore_init_args: Set[str] = (), **kwargs):
    init_params = _extract_init_args(cls, float_instance, ignore_init_args)
    init_params.update(**kwargs)
    return cls(**init_params)


class SGQuantMixin(nn.Module):
    """
    A base class for user custom Quantized classes.
    Every Quantized class must inherit this mixin, which adds `from_float` class-method.
    NOTES:
        * the Quantized class may also inherit from the native `QuantMixin` or `QuantInputMixin`
        * quant descriptors (for inputs and weights) will be passed as `kwargs`. The module may ignore them if they are
          not necessary
        * the default implementation of `from_float` is inspecting the __init__ args, and searching for corresponding
          properties from the float instance that is passed as argument, e.g., for `__init__(self, a)`
          the mechanism will look for `float_instance.a` and pass that value to the `__init__` method
    """

    @classmethod
    def from_float(cls, float_instance, **kwargs):
        required_init_params = list(inspect.signature(cls.__init__).parameters)[1:]  # [0] is self

        # if cls.__init__ has explicit `quant_desc_input` or `quant_desc_weight` - we don't search the state of the
        # float module, because it would not contain this state. these values are injected by the framework
        ignore_init_args = {"quant_desc_input", "quant_desc_weight"}.intersection(set(required_init_params))

        # if cls.__init__ doesn't have neither **kwargs, nor `quant_desc_input` and `quant_desc_weight`,
        # we should also remove these keys from the passed kwargs and make sure there's nothing more!
        if "kwargs" not in required_init_params:
            for arg in ("quant_desc_input", "quant_desc_weight"):
                if arg in ignore_init_args:
                    continue
                kwargs.pop(arg, None)  # we ignore if not existing

        return _from_float(cls, float_instance, ignore_init_args, **kwargs)


class SkipQuantization(nn.Module):
    """
    This class wraps a float module instance, and defines that this instance will not be converted to quantized version

    Example:
        self.my_block = SkipQuantization(MyBlock(4, n_classes))
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.float_module = module
        self.forward = module.forward


@dataclass(init=True)
class QuantizedMetadata:
    """
    This dataclass is responsible for holding the information regarding float->quantized module relation.
    It can be both layer-grained and module-grained, e.g.,
    `module.backbone.conv1 -> QuantConv2d`, `nn.Linear -> QuantLinear`, etc...

    :param float_source:          Name of a specific layer (e.g., `module.backbone.conv1`),
                                        or a specific type (e.g., `Conv2d`) that will be later quantized
    :param quantized_target_class: Quantized type that the source will be converted to
    :param action:                     how to resolve the conversion, we either:
                                - SKIP: skip it,
                                - UNWRAP: unwrap the instance and work with the wrapped one
                                  (i.e., we wrap with a mapper),
                                - REPLACE: replace source with an instance of the
                                  quantized type
                                - REPLACE_AND_RECURE: replace source with an instance of the
                                  quantized type, then try to recursively quantize the child modules of that type
                                - RECURE_AND_REPLACE: recursively quantize the child modules, then
                                  replace source with an instance of the quantized type
    :param input_quant_descriptor:     Quantization descriptor for inputs (None will take the default one)
    :param weights_quant_descriptor:   Quantization descriptor for weights (None will take the default one)
    """

    class ReplacementAction(Enum):
        REPLACE = "replace"
        REPLACE_AND_RECURE = "replace_and_recure"
        RECURE_AND_REPLACE = "recure_and_replace"
        UNWRAP = "unwrap"
        SKIP = "skip"

    float_source: Union[str, Type]
    quantized_target_class: Optional[Union[Type[QuantMixin], Type[QuantInputMixin], Type[SGQuantMixin]]]
    action: ReplacementAction
    input_quant_descriptor: QuantDescriptor = None  # default is used if None
    weights_quant_descriptor: QuantDescriptor = None  # default is used if None

    def __post_init__(self):
        if self.action in (
            QuantizedMetadata.ReplacementAction.REPLACE,
            QuantizedMetadata.ReplacementAction.REPLACE_AND_RECURE,
            QuantizedMetadata.ReplacementAction.RECURE_AND_REPLACE,
        ):
            assert issubclass(self.quantized_target_class, (SGQuantMixin, QuantMixin, QuantInputMixin))


class QuantizedMapping(nn.Module):
    """
    This class wraps a float module instance, and defines a mapping from this instance to the corresponding quantized
    class, with relevant quant descriptors.

    Example:
        self.my_block = QuantizedMapping(float_module=MyBlock(4, n_classes), quantized_target_class=MyQuantizedBlock)
    """

    def __init__(
        self,
        *,
        float_module: nn.Module,
        quantized_target_class: Union[Type[QuantMixin], Type[QuantInputMixin], Type[SGQuantMixin]],
        action=QuantizedMetadata.ReplacementAction.REPLACE,
        input_quant_descriptor: QuantDescriptor = None,
        weights_quant_descriptor: QuantDescriptor = None,
    ) -> None:
        super().__init__()
        self.float_module = float_module
        self.quantized_target_class = quantized_target_class
        self.action = action
        self.input_quant_descriptor = input_quant_descriptor
        self.weights_quant_descriptor = weights_quant_descriptor
        self.forward = float_module.forward


def _inject_class_methods_to_default_quant_types():
    """
    This is used to add `from_float` capability for the "native" pytorch-quantization (=nvidia-tensorrt) quant classes
    It allows SG to support these modules out of the box
    """
    import pytorch_quantization.quant_modules

    for quant_entry in pytorch_quantization.quant_modules._DEFAULT_QUANT_MAP:
        quant_cls = quant_entry.replace_mod
        quant_cls.from_float = classmethod(_from_float)
