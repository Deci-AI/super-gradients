import torch.optim as optim
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.factories.optimizers_type_factory import OptimizersTypeFactory
from super_gradients.training.params import (
    DEFAULT_OPTIMIZER_PARAMS_SGD,
    DEFAULT_OPTIMIZER_PARAMS_ADAM,
    DEFAULT_OPTIMIZER_PARAMS_RMSPROP,
    DEFAULT_OPTIMIZER_PARAMS_RMSPROPTF,
)
from super_gradients.training.utils import get_param
from super_gradients.training.utils.optimizers.rmsprop_tf import RMSpropTF

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
    batchnorm_types = (_BatchNorm,)
    torch_weight_with_bias_types = (_ConvNd, nn.Linear)
    no_decay_ids = []
    for name, m in module.named_modules():
        if isinstance(m, batchnorm_types):
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
    if isinstance(training_params.optimizer, str):
        optimizer_cls = OptimizersTypeFactory().get(training_params.optimizer)
    else:
        optimizer_cls = training_params.optimizer
    optimizer_params = OPTIMIZERS_DEFAULT_PARAMS[optimizer_cls].copy() if optimizer_cls in OPTIMIZERS_DEFAULT_PARAMS.keys() else dict()
    optimizer_params.update(**training_params.optimizer_params)
    training_params.optimizer_params = optimizer_params

    weight_decay = get_param(training_params.optimizer_params, "weight_decay", 0.0)
    # OPTIMIZER PARAM GROUPS ARE SET USING DEFAULT OR MODEL SPECIFIC INIT
    if hasattr(net.module, "initialize_param_groups"):
        # INITIALIZE_PARAM_GROUPS MUST RETURN A LIST OF DICTS WITH 'named_params' AND OPTIMIZER's ATTRIBUTES PER GROUP
        net_named_params = net.module.initialize_param_groups(lr, training_params)
    else:
        net_named_params = [{"named_params": net.named_parameters()}]

    if training_params.zero_weight_decay_on_bias_and_bn:
        optimizer_training_params = separate_zero_wd_params_groups_for_optimizer(net.module, net_named_params, weight_decay)

    else:
        # Overwrite groups to include params instead of named params
        for ind_group, param_group in enumerate(net_named_params):
            param_group["params"] = [param[1] for param in list(param_group["named_params"])]
            del param_group["named_params"]
            net_named_params[ind_group] = param_group
        optimizer_training_params = net_named_params

    # CREATE AN OPTIMIZER OBJECT AND INITIALIZE IT
    optimizer = optimizer_cls(optimizer_training_params, lr=lr, **training_params.optimizer_params)

    return optimizer
