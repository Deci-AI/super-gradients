import math
import warnings
from copy import deepcopy
from typing import Union

import torch
from torch import nn

from super_gradients.training.models import SgModule


def copy_attr(a: nn.Module, b: nn.Module, include: Union[list, tuple] = (), exclude: Union[list, tuple] = ()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay: float = 0.9999, beta: float = 15, exp_activation: bool = True):
        """
        Init the EMA
        :param model: Union[SgModule, nn.Module], the training model to construct the EMA model by
                    IMPORTANT: WHEN THE APPLICATION OF EMA ONLY ON A SUBSET OF ATTRIBUTES IS DESIRED, WRAP THE NN.MODULE
                    AS SgModule AND OVERWRITE get_include_attributes() AND get_exclude_attributes() AS DESIRED (SEE
                    YoLoV5Base IMPLEMENTATION IN super_gradients.trainer.models.yolov5.py AS AN EXAMPLE).
        :param decay: the maximum decay value. as the training process advances, the decay will climb towards this value
                      until the EMA_t+1 = EMA_t * decay + TRAINING_MODEL * (1- decay)
        :param beta: the exponent coefficient. The higher the beta, the sooner in the training the decay will saturate to
                     its final value. beta=15 is ~40% of the training process.
        """
        # Create EMA
        self.ema = deepcopy(model)
        self.ema.eval()
        if exp_activation:
            self.decay_function = lambda x: decay * (1 - math.exp(-x * beta))  # decay exponential ramp (to help early epochs)
        else:
            self.decay_function = lambda x: decay  # always return the same decay factor

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
            warnings.warn("Warning: EMA should be used with SgModule instance. All attributes of the model will be "
                          "included in EMA")
            self.include_attributes = []
            self.exclude_attributes = []
        for p in self.ema.module.parameters():
            p.requires_grad_(False)

    def update(self, model, training_percent: float):
        """
        Update the state of the EMA model.
        :param model: current training model
        :param training_percent: the percentage of the training process [0,1]. i.e 0.4 means 40% of the training have passed
        """
        # Update EMA parameters
        with torch.no_grad():
            decay = self.decay_function(training_percent)

            for ema_v, model_v in zip(self.ema.module.state_dict().values(), model.state_dict().values()):
                if ema_v.dtype.is_floating_point:
                    ema_v.copy_(ema_v * decay + (1. - decay) * model_v.detach())

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
