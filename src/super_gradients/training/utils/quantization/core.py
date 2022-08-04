import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Union, Type, Optional

from pytorch_quantization.nn.modules._utils import QuantMixin, QuantInputMixin
from pytorch_quantization.tensor_quant import QuantDescriptor
from torch import nn


def _extract_init_args(cls, float_instance):
    required_init_params = list(inspect.signature(cls.__init__).parameters)[1:]  # [0] is self

    if 'kwargs' in required_init_params:  # we don't want to search for a state named `kwargs`
        required_init_params.pop(required_init_params.index('kwargs'))

    float_instance_state = {}
    for p in required_init_params:
        if not hasattr(float_instance, p):
            raise ValueError(f"{float_instance.__class__.__name__} is missing `{p}` which is required "
                             f"in {cls.__name__}.__init__. Either override `SGQuantBase.from_float` "
                             f"or add {p} as state for {float_instance.__class__.__name__}.")
        float_instance_state[p] = getattr(float_instance, p)

    # Edge-cases
    if 'bias' in float_instance_state:
        if float_instance_state['bias'] is None:  # None is the state when bias=False in torch.nn
            float_instance_state['bias'] = False
        elif not isinstance(float_instance_state['bias'], bool):  # Tensor is the state when bias=True in torch.nn
            float_instance_state['bias'] = True
        # in case bias is a boolean - we don't do anything, so it is taken as-is, either True or False
    return float_instance_state


def _from_float(cls, float_instance, **kwargs):
    init_params = _extract_init_args(cls, float_instance)
    init_params.update(**kwargs)
    return cls(**init_params)


class SGQuantMixin(nn.Module):
    @classmethod
    def from_float(cls, float_instance, **kwargs):
        return _from_float(cls, float_instance, **kwargs)


class SkipQuantization(nn.Module):

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.float_module = module
        self.forward = module.forward


class QuantizedMapping(nn.Module):
    def __init__(self, module: nn.Module, quantized_type: Type[SGQuantMixin],
                 input_quant_descriptor: QuantDescriptor = None,
                 weights_quant_descriptor: QuantDescriptor = None) -> None:
        super().__init__()
        self.float_module = module
        self.quantized_type = quantized_type
        self.input_quant_descriptor = input_quant_descriptor,
        self.weights_quant_descriptor = weights_quant_descriptor,
        self.forward = module.forward


@dataclass(init=True)
class QuantizedMetadata:

    class ReplacementAction(Enum):
        REPLACE = 'replace'
        UNWRAP = 'unwrap'
        SKIP = 'skip'

    float_source: Union[str, Type]
    quantized_type: Optional[Union[Type[QuantMixin], Type[QuantInputMixin], Type[SGQuantMixin]]]
    action: ReplacementAction
    input_quant_descriptor: QuantDescriptor = None  # default is used if None
    weights_quant_descriptor: QuantDescriptor = None  # default is used if None

    def __post_init__(self):
        if self.action == QuantizedMetadata.ReplacementAction.REPLACE:
            assert issubclass(self.quantized_type, (SGQuantMixin, QuantMixin, QuantInputMixin))


def _inject_class_methods_to_default_quant_types():
    """
    This is used to add `from_float` capability for the "native" pytorch-quantization (=nvidia-tensorrt) quant classes
    """
    import pytorch_quantization.quant_modules
    for quant_entry in pytorch_quantization.quant_modules._DEFAULT_QUANT_MAP:
        quant_cls = quant_entry.replace_mod
        quant_cls.from_float = classmethod(_from_float)
