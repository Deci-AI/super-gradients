import inspect
import pytorch_quantization.nn as quant_nn
from pytorch_quantization.nn.modules._utils import QuantMixin, QuantInputMixin
from torch import nn


class AbstractSGQuant(nn.Module):
    @classmethod
    def _extract_init_args(cls, float_instance):
        required_init_params = list(inspect.signature(cls.__init__).parameters)[1:]  # [0] is self
        if 'kwargs' in required_init_params:
            required_init_params.pop(required_init_params.index('kwargs'))
        float_instance_state = {}
        for p in required_init_params:
            if not hasattr(float_instance, p):
                raise ValueError(f"{float_instance.__class__.__name__} is missing `{p}` which is required "
                                 f"in {cls.__name__}.__init__. Either override `SGQuantBase.from_float` "
                                 f"or add {p} as state for {float_instance.__class__.__name__}.")
            float_instance_state[p] = getattr(float_instance, p)
        if 'bias' in float_instance_state:
            float_instance_state['bias'] = float_instance_state['bias'] is not None
        return float_instance_state


class SGQuantInputAndWeights(AbstractSGQuant, QuantMixin):
    @classmethod
    def from_float(cls, float_instance, **kwargs):
        if 'quant_desc_input' not in kwargs:
            raise KeyError('`quant_desc_input` must appear in kwargs')
        if 'quant_desc_weight' not in kwargs:
            raise KeyError('`quant_desc_weight` must appear in kwargs')

        init_params = cls._extract_init_args(float_instance)
        init_params.update(**kwargs)
        return cls(**init_params)


class SGQuantInputOnly(AbstractSGQuant, QuantInputMixin):
    @classmethod
    def from_float(cls, float_instance, **kwargs):
        if 'quant_desc_input' not in kwargs:
            raise KeyError('`quant_desc_input` must appear in kwargs')
        if 'quant_desc_weight' in kwargs:
            raise KeyError('`quant_desc_weight` must NOT appear in kwargs')

        init_params = cls._extract_init_args(float_instance)
        init_params.update(**kwargs)
        return cls.__init__(**init_params)


class SkipQuantization(nn.Module):

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.float_module = module
        self.forward = module.forward


class SGTensorQuantizer(quant_nn.TensorQuantizer):
    pass


class SGQuantConv1d(quant_nn.QuantConv1d, SGQuantInputAndWeights):
    pass


class SGQuantConv2d(quant_nn.QuantConv2d, SGQuantInputAndWeights):
    pass


class SGQuantConv3d(quant_nn.QuantConv3d, SGQuantInputAndWeights):
    pass


class SGQuantConvTranspose1d(quant_nn.QuantConvTranspose1d, SGQuantInputAndWeights):
    pass


class SGQuantConvTranspose2d(quant_nn.QuantConvTranspose2d, SGQuantInputAndWeights):
    pass


class SGQuantConvTranspose3d(quant_nn.QuantConvTranspose3d, SGQuantInputAndWeights):
    pass


class SGQuantLinear(quant_nn.QuantLinear, SGQuantInputAndWeights):
    pass


class SGQuantLSTM(quant_nn.QuantLSTM, SGQuantInputAndWeights):
    pass


class SGQuantLSTMCell(quant_nn.QuantLSTMCell, SGQuantInputAndWeights):
    pass


class SGQuantAvgPool1d(quant_nn.QuantAvgPool1d, SGQuantInputOnly):
    pass


class SGQuantAvgPool2d(quant_nn.QuantAvgPool2d, SGQuantInputOnly):
    pass


class SGQuantAvgPool3d(quant_nn.QuantAvgPool3d, SGQuantInputOnly):
    pass


class SGQuantAdaptiveAvgPool1d(quant_nn.QuantAdaptiveAvgPool1d, SGQuantInputOnly):
    pass


class SGQuantAdaptiveAvgPool2d(quant_nn.QuantAdaptiveAvgPool2d, SGQuantInputOnly):
    pass


class SGQuantAdaptiveAvgPool3d(quant_nn.QuantAdaptiveAvgPool3d, SGQuantInputOnly):
    pass
