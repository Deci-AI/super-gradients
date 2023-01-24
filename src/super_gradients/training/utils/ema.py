import warnings
from copy import deepcopy
from typing import Union

import torch
from torch import nn

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.exceptions.factory_exceptions import UnknownTypeException
from super_gradients.training import utils as core_utils
from super_gradients.training.models import SgModule
from super_gradients.training.models.kd_modules.kd_module import KDModule
from super_gradients.training.utils.ema_decay_schedules import IDecayFunction, EMA_DECAY_FUNCTIONS

logger = get_logger(__name__)


def copy_attr(a: nn.Module, b: nn.Module, include: Union[list, tuple] = (), exclude: Union[list, tuple] = ()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay: float, decay_function: IDecayFunction):
        """
        Init the EMA
        :param model: Union[SgModule, nn.Module], the training model to construct the EMA model by
                    IMPORTANT: WHEN THE APPLICATION OF EMA ONLY ON A SUBSET OF ATTRIBUTES IS DESIRED, WRAP THE NN.MODULE
                    AS SgModule AND OVERWRITE get_include_attributes() AND get_exclude_attributes() AS DESIRED.
        :param decay: the maximum decay value. as the training process advances, the decay will climb towards this value
                      until the EMA_t+1 = EMA_t * decay + TRAINING_MODEL * (1- decay)
        :param beta: the exponent coefficient. The higher the beta, the sooner in the training the decay will saturate to
                     its final value. beta=15 is ~40% of the training process.
        """
        # Create EMA
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.decay_function = decay_function

        """"
        we hold a list of model attributes (not wights and biases) which we would like to include in each
        attribute update or exclude from each update. a SgModule declare these attribute using
        get_include_attributes and get_exclude_attributes functions. for a nn.Module which is not a SgModule
        all non-private (not starting with '_') attributes will be updated (and only them).
        """
        if isinstance(model.module, SgModule):
            self.include_attributes = model.module.get_include_attributes()
            self.exclude_attributes = model.module.get_exclude_attributes()
        else:
            warnings.warn("Warning: EMA should be used with SgModule instance. All attributes of the model will be " "included in EMA")
            self.include_attributes = []
            self.exclude_attributes = []
        for p in self.ema.module.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_params(cls, model: nn.Module, decay_type: str = None, decay: float = None, **kwargs):
        if decay is None:
            logger.warning(
                "Parameter `decay` is not specified for EMA params. Please specify `decay` parameter explicitly in your config:\n"
                "ema: True\n"
                "ema_params: \n"
                "  decay: 0.9999\n"
                "  decay_type: exp\n"
                "  beta: 15\n"
                "Will default to decay: 0.9999\n"
                "In the next major release of SG this warning will become an error."
            )
            decay = 0.9999

        if "exp_activation" in kwargs:
            logger.warning(
                "Parameter `exp_activation` is deprecated for EMA model. Please update your config to use decay_type: str (constant|exp|threshold) instead:\n"
                "ema: True\n"
                "ema_params: \n"
                "  decay: 0.9999\n"
                "  decay_type: exp # Equivalent to exp_activation: True\n"
                "  beta: 15\n"
                "\n"
                "ema: True\n"
                "ema_params: \n"
                "  decay: 0.9999\n"
                "  decay_type: constant # Equivalent to exp_activation: False\n"
                "\n"
                "In the next major release of SG this warning will become an error."
            )
            decay_type = "exp" if bool(kwargs.pop("exp_activation")) else "constant"

        if decay_type is None:
            logger.warning(
                "Parameter decay_type is not specified for EMA model. Please specify decay_type parameter explicitly in your config:\n"
                "ema: True\n"
                "ema_params: \n"
                "  decay: 0.9999\n"
                "  decay_type: constant|exp|threshold\n"
                "Will default to `exp` decay with beta = 15\n"
                "In the next major release of SG this warning will become an error."
            )
            decay_type = "exp"
            if "beta" not in kwargs:
                kwargs["beta"] = 15

        try:
            decay_cls = EMA_DECAY_FUNCTIONS[decay_type]
        except KeyError:
            raise UnknownTypeException(decay_type, list(EMA_DECAY_FUNCTIONS.keys()))

        decay_function = decay_cls(**kwargs)
        return cls(model, decay, decay_function)

    def update(self, model, step: int, total_steps: int):
        """
        Update the state of the EMA model.

        :param model: Current training model
        :param step: Current training step
        :param total_steps: Total training steps
        """
        # Update EMA parameters
        with torch.no_grad():
            decay = self.decay_function(self.decay, step, total_steps)

            for ema_v, model_v in zip(self.ema.module.state_dict().values(), model.state_dict().values()):
                if ema_v.dtype.is_floating_point:
                    ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v.detach())

    def update_attr(self, model):
        """
        This function updates model attributes (not weight and biases) from original model to the ema model.
        attributes of the original model, such as anchors and grids (of detection models), may be crucial to the
        model operation and need to be updated.
        If include_attributes and exclude_attributes lists were not defined, all non-private (not starting with '_')
        attributes will be updated (and only them).
        :param model: the source model
        """
        copy_attr(self.ema.module, model.module, self.include_attributes, self.exclude_attributes)


class KDModelEMA(ModelEMA):
    """Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, kd_model: KDModule, decay: float, decay_function: IDecayFunction):
        """
        Init the EMA
        :param kd_model: KDModule, the training Knowledge distillation model to construct the EMA model by
                    IMPORTANT: WHEN THE APPLICATION OF EMA ONLY ON A SUBSET OF ATTRIBUTES IS DESIRED, WRAP THE NN.MODULE
                    AS SgModule AND OVERWRITE get_include_attributes() AND get_exclude_attributes() AS DESIRED.
        :param decay: the maximum decay value. as the training process advances, the decay will climb towards this value
                      until the EMA_t+1 = EMA_t * decay + TRAINING_MODEL * (1- decay)
        :param beta: the exponent coefficient. The higher the beta, the sooner in the training the decay will saturate to
                     its final value. beta=15 is ~40% of the training process.
        """
        # Only work on the student (we don't want to update and to have a duplicate of the teacher)
        super().__init__(model=core_utils.WrappedModel(kd_model.module.student), decay=decay, decay_function=decay_function)

        # Overwrite current ema attribute with combination of the student model EMA (current self.ema)
        # with already the instantiated teacher, to have the final KD EMA
        self.ema = core_utils.WrappedModel(
            KDModule(
                arch_params=kd_model.module.arch_params,
                student=self.ema.module,
                teacher=kd_model.module.teacher,
                run_teacher_on_eval=kd_model.module.run_teacher_on_eval,
            )
        )
