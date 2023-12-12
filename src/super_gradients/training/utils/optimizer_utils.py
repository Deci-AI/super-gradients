import warnings

import torch.nn as nn
import torch.optim as optim
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.factories.optimizers_type_factory import OptimizersTypeFactory
from super_gradients.module_interfaces import SupportsFineTune
from super_gradients.training.params import (
    DEFAULT_OPTIMIZER_PARAMS_SGD,
    DEFAULT_OPTIMIZER_PARAMS_ADAM,
    DEFAULT_OPTIMIZER_PARAMS_RMSPROP,
    DEFAULT_OPTIMIZER_PARAMS_RMSPROPTF,
)
from super_gradients.training.utils import get_param
from super_gradients.training.utils.optimizers.rmsprop_tf import RMSpropTF
from super_gradients.training.utils.utils import is_model_wrapped
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from typing import List, Dict, Union
import torch

logger = get_logger(__name__)

OPTIMIZERS_DEFAULT_PARAMS = {
    optim.SGD: DEFAULT_OPTIMIZER_PARAMS_SGD,
    optim.Adam: DEFAULT_OPTIMIZER_PARAMS_ADAM,
    optim.RMSprop: DEFAULT_OPTIMIZER_PARAMS_RMSPROP,
    RMSpropTF: DEFAULT_OPTIMIZER_PARAMS_RMSPROPTF,
}


def separate_zero_wd_params_groups_for_optimizer(module: nn.Module, net_named_params, weight_decay: float):
    """
    separate param groups for batchnorm and biases and others with weight decay. return list of param groups in format
     required by torch Optimizer classes.
    bias + BN with weight decay=0 and the rest with the given weight decay
        :param module: train net module.
        :param net_named_params: list of params groups, output of SgModule.initialize_param_groups
        :param weight_decay: value to set for the non BN and bias parameters
    """
    # FIXME - replace usage of ids addresses to find batchnorm and biases params.
    #  This solution iterate 2 times over module parameters, find a way to iterate only one time.
    no_decay_ids = _get_no_decay_param_ids(module)
    # split param groups for optimizer
    optimizer_param_groups = []
    for param_group in net_named_params:
        no_decay_params = []
        decay_params = []
        for name, param in param_group["named_params"]:
            if id(param) in no_decay_ids:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        # append two param groups from the original param group, with and without weight decay.
        extra_optim_params = {key: param_group[key] for key in param_group if key not in ["named_params", "weight_decay"]}
        optimizer_param_groups.append({"params": no_decay_params, "weight_decay": 0.0, **extra_optim_params})
        optimizer_param_groups.append({"params": decay_params, "weight_decay": weight_decay, **extra_optim_params})

    return optimizer_param_groups


def _get_no_decay_param_ids(module: nn.Module):
    # FIXME - replace usage of ids addresses to find batchnorm and biases params.
    #  Use other common way to identify torch parameters other than id or layer names
    """
    Iterate over module.modules() and returns params id addresses of batch-norm and biases params.
    NOTE - ALL MODULES WITH ATTRIBUTES NAMED BIAS AND ARE INSTANCE OF nn.Parameter WILL BE CONSIDERED A BIAS PARAM FOR
        ZERO WEIGHT DECAY.
    """
    norm_types = (_BatchNorm, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    torch_weight_with_bias_types = (_ConvNd, nn.Linear)
    no_decay_ids = []
    for name, m in module.named_modules():
        if isinstance(m, norm_types):
            no_decay_ids.append(id(m.weight))
            no_decay_ids.append(id(m.bias))
        elif hasattr(m, "bias") and isinstance(m.bias, nn.Parameter):
            if not isinstance(m, torch_weight_with_bias_types):
                logger.warning(
                    f"Module class: {m.__class__}, have a `bias` parameter attribute but is not instance of"
                    f" torch primitive modules, this bias parameter will be part of param group with zero"
                    f" weight decay."
                )
            no_decay_ids.append(id(m.bias))
    return no_decay_ids


def build_optimizer(net: nn.Module, lr: float, training_params) -> optim.Optimizer:
    """
    Wrapper function for initializing the optimizer
        :param net: the nn_module to build the optimizer for
        :param lr: initial learning rate
        :param training_params: training_parameters
    """
    if is_model_wrapped(net):
        raise ValueError("Argument net for build_optimizer must be an unwrapped model. " "Please use build_optimizer(unwrap_model(net), ...).")
    if isinstance(training_params.optimizer, str):
        optimizer_cls = OptimizersTypeFactory().get(training_params.optimizer)
    else:
        optimizer_cls = training_params.optimizer
    optimizer_params = OPTIMIZERS_DEFAULT_PARAMS[optimizer_cls].copy() if optimizer_cls in OPTIMIZERS_DEFAULT_PARAMS.keys() else dict()
    optimizer_params.update(**training_params.optimizer_params)
    training_params.optimizer_params = optimizer_params

    weight_decay = get_param(training_params.optimizer_params, "weight_decay", 0.0)
    # OPTIMIZER PARAM GROUPS ARE SET USING DEFAULT OR MODEL SPECIFIC INIT
    if hasattr(net, "initialize_param_groups") or hasattr(net, "update_param_groups"):
        warnings.warn(
            "initialize_param_groups and update_param_groups usages are deprecated since 3.4.0, will be removed in "
            "3.5.0 and have no effect. \n "
            "Assign different learning rates by passing a mapping of layer name prefixes to lr values through "
            "initial_lr training hyperparameter (i.e initial_lr={'backbone': 0.01, 'default':0.1})",
            DeprecationWarning,
        )
    if training_params.finetune:
        if not isinstance(net, SupportsFineTune):
            warnings.warn(
                "training hyperparameter finetune=True but will have no effect. get_finetune_lr_dict is not implemented for this model, which is required."
            )
        elif not isinstance(lr, float):
            raise RuntimeError("When training with fine_tune=True, initial_lr must be a scalar.")
        lr = net.get_finetune_lr_dict(lr)
        logger.info(f"Training with finetune=True: setting initial_lr to predefined mapping {lr}")
        training_params.initial_lr = lr

    net_named_params = initialize_param_groups(net, lr)

    if training_params.zero_weight_decay_on_bias_and_bn:
        optimizer_training_params = separate_zero_wd_params_groups_for_optimizer(net, net_named_params, weight_decay)

    else:
        # Overwrite groups to include params instead of named params
        for ind_group, param_group in enumerate(net_named_params):
            param_group["params"] = [param[1] for param in list(param_group["named_params"])]
            del param_group["named_params"]
            net_named_params[ind_group] = param_group
        optimizer_training_params = net_named_params

    # CREATE AN OPTIMIZER OBJECT AND INITIALIZE IT
    optimizer = optimizer_cls(optimizer_training_params, **training_params.optimizer_params)

    return optimizer


def separate_lr_groups(model: nn.Module, lr_dict: Dict[str, float]) -> List[Dict]:
    """
    Separate parameters based on specified learning rates for each group in the model.
    :param model: nn.Module model.
    :param lr_dict: Dictionary where keys are group names and values are the learning rates.
    :return: List of param groups with named_parameters and corresponding learning rates.
    """
    param_groups = []
    default_lr = lr_dict.get("default", None)
    if default_lr is None:
        raise RuntimeError("When passing initial_lr as dictionary, must pass 'default'.")
    group_names = set(lr_dict.keys()) - {"default"}

    for group_name in group_names:
        lr = lr_dict[group_name]
        named_params = [(name, param) for name, param in model.named_parameters() if name.startswith(group_name)]

        if lr == 0:
            for name, param in named_params:
                param.requires_grad = False  # Freeze the layer
        else:
            param_groups.append({"named_params": named_params, "lr": lr, "name": group_name})

    default_named_params = [
        (name, param) for name, param in model.named_parameters() if all(name.startswith(group) is False for group in group_names) and param.requires_grad
    ]
    if default_named_params:
        if default_lr != 0:
            param_groups.append({"named_params": default_named_params, "lr": default_lr, "name": "default"})
        else:
            for name, param in default_named_params:
                param.requires_grad = False  # Freeze the layer

    return param_groups


def initialize_param_groups(model: nn.Module, lr: Union[float, Dict[str, float]]) -> List[Dict]:
    """
    Custom param groups for training with specified learning rates for each group in the model.
    :param model: nn.Module model.
    :param lr: Dictionary where keys are group names and values are the learning rates,
     or a learning rate value when passed as a scalar.
    :return: List of param groups.
    """
    if isinstance(lr, float) or isinstance(lr, int):
        model_named_params = [{"named_params": model.named_parameters(), "lr": lr, "name": "default"}]
    else:
        model_named_params = separate_lr_groups(model, lr)
    return model_named_params


def name_optimizer_param_groups_inplace(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """
    Convert an optimizer's param_groups to use named parameters, modifying it in place.

    :param optimizer: torch.optim.Optimizer, The optimizer to be converted.

    Returns:
        torch.optim.Optimizer: The same optimizer with modified param_groups.
    """

    named_parameters = list(optimizer.param_groups[0]["params"])
    num_param_groups = len(optimizer.param_groups)
    group_name = [f"group_{i}" for i in range(num_param_groups)] if num_param_groups > 1 else "default"

    for i, param_group in enumerate(optimizer.param_groups):
        param_group["params"] = named_parameters
        param_group["name"] = group_name if num_param_groups == 1 else group_name[i]

    return optimizer


def get_initial_lr_from_optimizer(optimizer: torch.optim.Optimizer) -> Union[Dict[str, float], float]:
    """
    Returns Initial learning rate as:

    float - learning rate value when passed as a scalar
    Dictionary where keys are group names and values are the learning rates.
    For example {"default": 0.01, "head": 0.1}

    Does so by iterating over the optmizer.param_groups and extracting the "lr" vaules.
    If the optimizer was intiialized with .parameters() and not named_paramters(), names will be assigned to the
     optimizer parameter groups by index.

    :param optimizer: torch.optim.Optimizer, The optimizer to extract the lrs from.
    :return: initial_lr as described above.
    """
    if "name" not in optimizer.param_groups[0].keys():
        optimizer = name_optimizer_param_groups_inplace(optimizer)
    if len(optimizer.param_groups) == 1:
        initial_lr = optimizer.param_groups[0]["lr"]
    else:
        initial_lr = {group["name"]: group["lr"] for group in optimizer.param_groups}
    return initial_lr
