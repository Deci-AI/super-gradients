from pathlib import Path
from typing import Tuple, Type, Optional, Union

import hydra
import torch

from super_gradients.common.data_types.enum.strict_load import StrictLoad
from super_gradients.common.plugins.deci_client import DeciClient, client_enabled
from super_gradients.module_interfaces import HasPredict
from super_gradients.training import utils as core_utils
from super_gradients.common.exceptions.factory_exceptions import UnknownTypeException
from super_gradients.training.models import SgModule
from super_gradients.common.registry.registry import ARCHITECTURES
from super_gradients.training.pretrained_models import PRETRAINED_NUM_CLASSES
from super_gradients.training.utils import HpmStruct, get_param
from super_gradients.training.utils.checkpoint_utils import (
    load_checkpoint_to_model,
    load_pretrained_weights,
    read_ckpt_state_dict,
    load_pretrained_weights_local,
)
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.sg_trainer_utils import get_callable_param_names
from super_gradients.training.processing.processing import get_pretrained_processing_params

logger = get_logger(__name__)


def get_architecture(model_name: str, arch_params: HpmStruct, download_required_code: bool = True) -> Tuple[Type[torch.nn.Module], HpmStruct, str, bool]:
    """
    Get the corresponding architecture class.

    :param model_name:          Define the model's architecture from models/ALL_ARCHITECTURES
    :param arch_params:         Architecture hyper parameters. e.g.: block, num_blocks, etc.
    :param download_required_code: if model is not found in SG and is downloaded from a remote client, overriding this parameter with False
                                        will prevent additional code from being downloaded. This affects only models from remote client.

    :return:
        - architecture_cls:     Class of the model
        - arch_params:          Might be updated if loading from remote deci lab
        - pretrained_weights_path:   path to the pretrained weights from deci lab (None for local models).
        - is_remote:            True if loading from remote deci lab
    """
    pretrained_weights_path = None
    is_remote = False
    if not isinstance(model_name, str):
        raise ValueError("Parameter model_name is expected to be a string.")

    architecture = get_param(ARCHITECTURES, model_name)
    if model_name not in ARCHITECTURES.keys() and architecture is None:
        if client_enabled:
            logger.info(f'The required model, "{model_name}", was not found in SuperGradients. Trying to load a model from remote deci-lab')
            deci_client = DeciClient()

            _arch_params = deci_client.get_model_arch_params(model_name)
            if _arch_params is None:
                raise ValueError(
                    f'The required model "{model_name}", was not found in SuperGradients and remote deci-lab. '
                    f"See docs or all_architectures.py for supported model names."
                )

            if download_required_code:  # Some extra code might be required to instantiate the arch params.
                deci_client.download_and_load_model_additional_code(model_name, target_path=str(Path.cwd()))
            _arch_params = hydra.utils.instantiate(_arch_params)

            pretrained_weights_path = deci_client.get_model_weights(model_name)
            model_name = _arch_params["model_name"]
            del _arch_params["model_name"]
            _arch_params = HpmStruct(**_arch_params)
            _arch_params.override(**arch_params.to_dict())
            arch_params, is_remote = _arch_params, True
        else:
            raise UnknownTypeException(
                message=f'The required model, "{model_name}", was not found in SuperGradients. See docs or all_architectures.py for supported model names.',
                unknown_type=model_name,
                choices=list(ARCHITECTURES.keys()),
            )

    return get_param(ARCHITECTURES, model_name), arch_params, pretrained_weights_path, is_remote


def instantiate_model(
    model_name: str, arch_params: dict, num_classes: int, pretrained_weights: str = None, download_required_code: bool = True
) -> Union[SgModule, torch.nn.Module]:
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

    architecture_cls, arch_params, pretrained_weights_path, is_remote = get_architecture(model_name, arch_params, download_required_code)

    if not issubclass(architecture_cls, SgModule):
        net = architecture_cls(**arch_params.to_dict(include_schema=False))
    else:
        if core_utils.get_param(arch_params, "num_classes"):
            logger.warning(
                "Passing num_classes through arch_params is deprecated and will be removed in the next version. " "Pass num_classes explicitly to models.get"
            )
            num_classes = num_classes or arch_params.num_classes

        if num_classes is not None:
            arch_params.override(num_classes=num_classes)

        if pretrained_weights is None and num_classes is None:
            raise ValueError("num_classes or pretrained_weights must be passed to determine net's structure.")

        if pretrained_weights:
            num_classes_new_head = core_utils.get_param(arch_params, "num_classes", PRETRAINED_NUM_CLASSES[pretrained_weights])
            arch_params.num_classes = PRETRAINED_NUM_CLASSES[pretrained_weights]

        # Most of the SG models work with a single params names "arch_params" of type HpmStruct, but a few take **kwargs instead
        if "arch_params" not in get_callable_param_names(architecture_cls):
            net = architecture_cls(**arch_params.to_dict(include_schema=False))
        else:
            net = architecture_cls(arch_params=arch_params)

        if pretrained_weights:
            if is_remote:
                load_pretrained_weights_local(net, model_name, pretrained_weights_path)
            else:
                load_pretrained_weights(net, model_name, pretrained_weights)

            if num_classes_new_head != arch_params.num_classes:
                net.replace_head(new_num_classes=num_classes_new_head)
                arch_params.num_classes = num_classes_new_head

            # STILL NEED TO GET PREPROCESSING PARAMS IN CASE CHECKPOINT HAS NO RECIPE
            if isinstance(net, HasPredict):
                processing_params = get_pretrained_processing_params(model_name, pretrained_weights)
                net.set_dataset_processing_params(**processing_params)

    _add_model_name_attribute(net, model_name)

    return net


def _add_model_name_attribute(model: torch.nn.Module, model_name: str) -> None:
    """Add an attribute to a model.
    This is useful to keep track of the exact name used to instantiate the model using `models.get()`,
    which differs to the class name because the same class can be used to build different architectures."""
    setattr(model, "_sg_model_name", model_name)


def get_model_name(model: torch.nn.Module) -> Optional[str]:
    """Get the name of a model loaded by SuperGradients' `models.get()`. If the model was not loaded using `models.get()`, return None."""
    return getattr(model, "_sg_model_name", None)


def get(
    model_name: str,
    arch_params: Optional[dict] = None,
    num_classes: int = None,
    strict_load: StrictLoad = StrictLoad.NO_KEY_MATCHING,
    checkpoint_path: str = None,
    pretrained_weights: str = None,
    load_backbone: bool = False,
    download_required_code: bool = True,
    checkpoint_num_classes: int = None,
) -> Union[SgModule, torch.nn.Module]:
    """
    :param model_name:          Defines the model's architecture from models/ALL_ARCHITECTURES
    :param arch_params:         Architecture hyper parameters. e.g.: block, num_blocks, etc.
    :param num_classes:         Number of classes (defines the net's structure).
                                    If None is given, will try to derrive from pretrained_weight's corresponding dataset.
    :param strict_load:         See super_gradients.common.data_types.enum.strict_load.StrictLoad class documentation for details
                                    (default=NO_KEY_MATCHING to suport SG trained checkpoints)
    :param checkpoint_path:     The path to the external checkpoint to be loaded. Can be absolute or relative (ie: path/to/checkpoint.pth) path or URL.
                                    If provided, will automatically attempt to load the checkpoint.
    :param pretrained_weights:  Describe the dataset of the pretrained weights (for example "imagenent").
    :param load_backbone:       Load the provided checkpoint to model.backbone instead of model.
    :param download_required_code: if model is not found in SG and is downloaded from a remote client, overriding this parameter with False
                                    will prevent additional code from being downloaded. This affects only models from remote client.
    :param checkpoint_num_classes:  num_classes of checkpoint_path/ pretrained_weights, when checkpoint_path is not None.
     Used when num_classes != checkpoint_num_class. In this case, the module will be initialized with checkpoint_num_class, then weights will be loaded. Finaly
        replace_head(new_num_classes=num_classes) is called (useful when wanting to perform transfer learning, from a checkpoint outside of
         then ones offered in SG model zoo).


    NOTE: Passing pretrained_weights and checkpoint_path is ill-defined and will raise an error.
    """
    checkpoint_num_classes = checkpoint_num_classes or num_classes

    if checkpoint_num_classes:
        net = instantiate_model(model_name, arch_params, checkpoint_num_classes, pretrained_weights, download_required_code)
    else:
        net = instantiate_model(model_name, arch_params, num_classes, pretrained_weights, download_required_code)

    if load_backbone and not checkpoint_path:
        raise ValueError("Please set checkpoint_path when load_backbone=True")

    if checkpoint_path:
        ckpt_entries = read_ckpt_state_dict(ckpt_path=checkpoint_path).keys()
        load_processing = "processing_params" in ckpt_entries
        load_ema_as_net = "ema_net" in ckpt_entries
        _ = load_checkpoint_to_model(
            ckpt_local_path=checkpoint_path,
            load_backbone=load_backbone,
            net=net,
            strict=strict_load.value if hasattr(strict_load, "value") else strict_load,
            load_weights_only=True,
            load_ema_as_net=load_ema_as_net,
            load_processing_params=load_processing,
        )
    if checkpoint_num_classes != num_classes:
        net.replace_head(new_num_classes=num_classes)

    return net
