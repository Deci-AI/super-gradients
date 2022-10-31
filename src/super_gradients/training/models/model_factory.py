from pathlib import Path
from typing import Tuple, Type, Optional

import hydra
import torch

from super_gradients.common import StrictLoad
from super_gradients.common.plugins.deci_client import DeciClient
from super_gradients.training import utils as core_utils
from super_gradients.training.models import SgModule
from super_gradients.training.models.all_architectures import ARCHITECTURES
from super_gradients.training.pretrained_models import PRETRAINED_NUM_CLASSES
from super_gradients.training.utils import HpmStruct
from super_gradients.training.utils.checkpoint_utils import (
    load_checkpoint_to_model,
    load_pretrained_weights,
    read_ckpt_state_dict,
    load_pretrained_weights_local,
)
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.sg_trainer_utils import get_callable_param_names

logger = get_logger(__name__)


def get_architecture(model_name: str, arch_params: HpmStruct, pretrained_weights: str,
                     download_required_code: bool = True) -> Tuple[Type[torch.nn.Module], HpmStruct, str, bool]:
    """
    Get the corresponding architecture class.

    :param model_name:          Define the model's architecture from models/ALL_ARCHITECTURES
    :param arch_params:         Architecture hyper parameters. e.g.: block, num_blocks, etc.
    :param pretrained_weights:  Describe the dataset of the pretrained weights (for example "imagenent")
    :param download_required_code: if model is not found in SG and is downloaded from a remote client, overriding this parameter with False
                                        will prevent additional code from being downloaded. This affects only models from remote client.

    :return:
        - architecture_cls:     Class of the model
        - arch_params:          Might be updated if loading from remote deci lab
        - pretrained_weights:   Might be updated if loading from remote deci lab
        - is_remote:            True if loading from remote deci lab
    """
    is_remote = False
    if not isinstance(model_name, str):
        raise ValueError("Parameter model_name is expected to be a string.")
    elif model_name not in ARCHITECTURES.keys():
        logger.info(f'Required model {model_name} not found in local SuperGradients. Trying to load a model from remote deci lab')
        deci_client = DeciClient()
        _arch_params = deci_client.get_model_arch_params(model_name)
        if download_required_code:
            deci_client.download_and_load_model_additional_code(model_name, Path.cwd())

        if _arch_params is None:
            raise ValueError("Unsupported model name " + str(model_name) + ", see docs or all_architectures.py for supported nets.")
        _arch_params = hydra.utils.instantiate(_arch_params)
        pretrained_weights = deci_client.get_model_weights(model_name)
        model_name = _arch_params['model_name']
        del _arch_params['model_name']
        _arch_params = HpmStruct(**_arch_params)
        _arch_params.override(**arch_params.to_dict())
        arch_params, is_remote = _arch_params, True
    return ARCHITECTURES[model_name], arch_params, pretrained_weights, is_remote


def instantiate_model(model_name: str, arch_params: dict, num_classes: int,
                      pretrained_weights: str = None, download_required_code: bool = True) -> torch.nn.Module:
    """
    Instantiates nn.Module according to architecture and arch_params, and handles pretrained weights and the required
        module manipulation (i.e head replacement).

    :param model_name:          Define the model's architecture from models/ALL_ARCHITECTURES
    :param arch_params:         Architecture hyper parameters. e.g.: block, num_blocks, etc.
    :param num_classes:         Number of classes (defines the net's structure).
                                    If None is given, will try to derrive from pretrained_weight's corresponding dataset.
    :param pretrained_weights:  Describe the dataset of the pretrained weights (for example "imagenent")
    :param download_required_code: if model is not found in SG and is downloaded from a remote client, overriding this parameter with False
                                will prevent additional code from being downloaded. This affects only models from remote client.

    :return:                    Instantiated model i.e torch.nn.Module, architecture_class (will be none when architecture is not str)
    """
    if arch_params is None:
        arch_params = {}
    arch_params = core_utils.HpmStruct(**arch_params)

    architecture_cls, arch_params, pretrained_weights, is_remote = get_architecture(model_name, arch_params, pretrained_weights, download_required_code)

    if not issubclass(architecture_cls, SgModule):
        net = architecture_cls(**arch_params.to_dict(include_schema=False))
    else:
        if core_utils.get_param(arch_params, "num_classes"):
            logger.warning("Passing num_classes through arch_params is deprecated and will be removed in the next version. "
                           "Pass num_classes explicitly to models.get")
            num_classes = arch_params.num_classes

        if num_classes is not None:
            arch_params.override(num_classes=num_classes)

        if pretrained_weights is None and num_classes is None:
            raise ValueError("num_classes or pretrained_weights must be passed to determine net's structure.")

        if pretrained_weights and not is_remote:
            num_classes_new_head = core_utils.get_param(arch_params, "num_classes", PRETRAINED_NUM_CLASSES[pretrained_weights])
            arch_params.num_classes = PRETRAINED_NUM_CLASSES[pretrained_weights]

        # Most of the SG models work with a single params names "arch_params" of type HpmStruct, but a few take **kwargs instead
        if "arch_params" not in get_callable_param_names(architecture_cls):
            net = architecture_cls(**arch_params.to_dict(include_schema=False))
        else:
            net = architecture_cls(arch_params=arch_params)

        if pretrained_weights:
            if is_remote:
                load_pretrained_weights_local(net, model_name, pretrained_weights)
            else:
                load_pretrained_weights(net, model_name, pretrained_weights)
                if num_classes_new_head != arch_params.num_classes:
                    net.replace_head(new_num_classes=num_classes_new_head)
                    arch_params.num_classes = num_classes_new_head
    return net


def get(model_name: str, arch_params: Optional[dict] = None, num_classes: int = None,
        strict_load: StrictLoad = StrictLoad.NO_KEY_MATCHING, checkpoint_path: str = None,
        pretrained_weights: str = None, load_backbone: bool = False, download_required_code: bool = True) -> SgModule:
    """
    :param model_name:          Defines the model's architecture from models/ALL_ARCHITECTURES
    :param arch_params:         Architecture hyper parameters. e.g.: block, num_blocks, etc.
    :param num_classes:         Number of classes (defines the net's structure).
                                    If None is given, will try to derrive from pretrained_weight's corresponding dataset.
    :param strict_load:         See super_gradients.common.data_types.enum.strict_load.StrictLoad class documentation for details
                                    (default=NO_KEY_MATCHING to suport SG trained checkpoints)
    :param checkpoint_path:     The path to the external checkpoint to be loaded. Can be absolute or relative (ie: path/to/checkpoint.pth).
                                    If provided, will automatically attempt to load the checkpoint.
    :param pretrained_weights:  Describe the dataset of the pretrained weights (for example "imagenent").
    :param load_backbone:       Load the provided checkpoint to model.backbone instead of model.
    :param download_required_code: if model is not found in SG and is downloaded from a remote client, overriding this parameter with False
                                    will prevent additional code from being downloaded. This affects only models from remote client.

    NOTE: Passing pretrained_weights and checkpoint_path is ill-defined and will raise an error.
    """

    net = instantiate_model(model_name, arch_params, num_classes, pretrained_weights, download_required_code)

    if load_backbone and not checkpoint_path:
        raise ValueError("Please set checkpoint_path when load_backbone=True")

    if checkpoint_path:
        load_ema_as_net = 'ema_net' in read_ckpt_state_dict(ckpt_path=checkpoint_path).keys()
        _ = load_checkpoint_to_model(ckpt_local_path=checkpoint_path,
                                     load_backbone=load_backbone,
                                     net=net,
                                     strict=strict_load.value if hasattr(strict_load, "value") else strict_load,
                                     load_weights_only=True,
                                     load_ema_as_net=load_ema_as_net)
    return net
