from functools import partial
from typing import Type, Union, Dict

import torch
from torch import nn


def get_activation_type(activation: Union[str, None], **kwargs) -> Type:
    """
    Returns activation function class by its name. This function support all modules available in torch.nn
    and also their lower-case aliases. On top of that, it supports a few aliaes: leaky_relu (LeakyReLU), swish (silu).

    >>> act_cls = get_activation_type("LeakyReLU", inplace=True, slope=0.01)
    >>> act = act_cls()

    Args:
        activation: Activation function name (E.g. ReLU)
        **kwargs: Extra arguments to pass to constructor during instantiation (E.g. inplace=True)

    Returns:
        Type of the activation function that is ready to be instantiated
    """

    if activation is None:
        activation_cls = nn.Identity
    else:
        lowercase_aliases: Dict[str, str] = dict((k.lower(), k) for k in torch.nn.__dict__.keys())

        # Register additional aliases
        lowercase_aliases["leaky_relu"] = "LeakyReLU"  # LeakyRelu in snake_case
        lowercase_aliases["swish"] = "SiLU"  # Swish shich is equivalent to SiLU

        if activation in lowercase_aliases:
            activation = lowercase_aliases[activation]

        activation_cls = torch.nn.__dict__[activation]
        if len(kwargs):
            activation_cls = partial(activation_cls, **kwargs)

    return activation_cls
