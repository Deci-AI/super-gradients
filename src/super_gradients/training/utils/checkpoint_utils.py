import collections
import os
import tempfile
from typing import Union, Mapping

import pkg_resources
import torch
from torch import nn, Tensor

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.data_interface.adnn_model_repository_data_interface import ADNNModelRepositoryDataInterfaces
from super_gradients.common.data_types import StrictLoad
from super_gradients.common.decorators.explicit_params_validator import explicit_params_validation
from super_gradients.module_interfaces import HasPredict
from super_gradients.training.pretrained_models import MODEL_URLS
from super_gradients.training.utils.distributed_training_utils import get_local_rank, wait_for_the_master

try:
    from torch.hub import download_url_to_file, load_state_dict_from_url
except (ModuleNotFoundError, ImportError, NameError):
    from torch.hub import _download_url_to_file as download_url_to_file


logger = get_logger(__name__)


def transfer_weights(model: nn.Module, model_state_dict: Mapping[str, Tensor]) -> None:
    """
    Copy weights from `model_state_dict` to `model`, skipping layers that are incompatible (Having different shape).
    This method is helpful if you are doing some model surgery and want to load
    part of the model weights into different model.
    This function will go over all the layers in `model_state_dict` and will try to find a matching layer in `model` and
    copy the weights into it. If shape will not match, the layer will be skipped.

    :param model: Model to load weights into
    :param model_state_dict: Model state dict to load weights from
    :return: None
    """
    for name, value in model_state_dict.items():
        try:
            model.load_state_dict(collections.OrderedDict([(name, value)]), strict=False)
        except RuntimeError:
            pass


def adaptive_load_state_dict(net: torch.nn.Module, state_dict: dict, strict: Union[bool, StrictLoad], solver=None):
    """
    Adaptively loads state_dict to net, by adapting the state_dict to net's layer names first.
    :param net: (nn.Module) to load state_dict to
    :param state_dict: (dict) Chekpoint state_dict
    :param strict: (StrictLoad) key matching strictness
    :param solver: callable with signature (ckpt_key, ckpt_val, model_key, model_val)
                     that returns a desired weight for ckpt_val.
    :return:
    """
    state_dict = state_dict["net"] if "net" in state_dict else state_dict
    try:
        strict_bool = strict if isinstance(strict, bool) else strict != StrictLoad.OFF
        net.load_state_dict(state_dict, strict=strict_bool)
    except (RuntimeError, ValueError, KeyError) as ex:
        if strict == StrictLoad.NO_KEY_MATCHING:
            adapted_state_dict = adapt_state_dict_to_fit_model_layer_names(net.state_dict(), state_dict, solver=solver)
            net.load_state_dict(adapted_state_dict["net"], strict=True)
        elif strict == StrictLoad.KEY_MATCHING:
            transfer_weights(net, state_dict)
        else:
            raise_informative_runtime_error(net.state_dict(), state_dict, ex)


@explicit_params_validation(validation_type="None")
def copy_ckpt_to_local_folder(
    local_ckpt_destination_dir: str,
    ckpt_filename: str,
    remote_ckpt_source_dir: str = None,
    path_src: str = "local",
    overwrite_local_ckpt: bool = False,
    load_weights_only: bool = False,
):
    """
    Copy the checkpoint from any supported source to a local destination path
        :param local_ckpt_destination_dir:  destination where the checkpoint will be saved to
        :param ckpt_filename:         ckpt_best.pth Or ckpt_latest.pth
        :param remote_ckpt_source_dir:       Name of the source checkpoint to be loaded (S3 Model\full URL)
        :param path_src:              S3 / url
        :param overwrite_local_ckpt:  determines if checkpoint will be saved in destination dir or in a temp folder

        :return: Path to checkpoint
    """
    ckpt_file_full_local_path = None

    # IF NOT DEFINED - IT IS SET TO THE TARGET's FOLDER NAME
    remote_ckpt_source_dir = local_ckpt_destination_dir if remote_ckpt_source_dir is None else remote_ckpt_source_dir

    if not overwrite_local_ckpt:
        # CREATE A TEMP FOLDER TO SAVE THE CHECKPOINT TO
        download_ckpt_destination_dir = tempfile.gettempdir()
        print(
            "PLEASE NOTICE - YOU ARE IMPORTING A REMOTE CHECKPOINT WITH overwrite_local_checkpoint = False "
            "-> IT WILL BE REDIRECTED TO A TEMP FOLDER AND DELETED ON MACHINE RESTART"
        )
    else:
        # SAVE THE CHECKPOINT TO MODEL's FOLDER
        download_ckpt_destination_dir = pkg_resources.resource_filename("checkpoints", local_ckpt_destination_dir)

    if path_src.startswith("s3"):
        model_checkpoints_data_interface = ADNNModelRepositoryDataInterfaces(data_connection_location=path_src)
        # DOWNLOAD THE FILE FROM S3 TO THE DESTINATION FOLDER
        ckpt_file_full_local_path = model_checkpoints_data_interface.load_remote_checkpoints_file(
            ckpt_source_remote_dir=remote_ckpt_source_dir,
            ckpt_destination_local_dir=download_ckpt_destination_dir,
            ckpt_file_name=ckpt_filename,
            overwrite_local_checkpoints_file=overwrite_local_ckpt,
        )

        if not load_weights_only:
            # COPY LOG FILES FROM THE REMOTE DIRECTORY TO THE LOCAL ONE ONLY IF LOADING THE CURRENT MODELs CKPT
            model_checkpoints_data_interface.load_all_remote_log_files(
                model_name=remote_ckpt_source_dir, model_checkpoint_local_dir=download_ckpt_destination_dir
            )

    if path_src == "url":
        ckpt_file_full_local_path = download_ckpt_destination_dir + os.path.sep + ckpt_filename
        # DOWNLOAD THE FILE FROM URL TO THE DESTINATION FOLDER
        with wait_for_the_master(get_local_rank()):
            download_url_to_file(remote_ckpt_source_dir, ckpt_file_full_local_path, progress=True)

    return ckpt_file_full_local_path


def read_ckpt_state_dict(ckpt_path: str, device="cpu") -> Mapping[str, torch.Tensor]:
    """
    Reads a checkpoint state dict from a given path or url

    :param ckpt_path: Checkpoint path or url
    :param device: Target devide where tensors should be loaded
    :return: Checkpoint state dict object
    """

    if ckpt_path.startswith("https://"):
        with wait_for_the_master(get_local_rank()):
            state_dict = load_state_dict_from_url(ckpt_path, progress=False, map_location=device)
        return state_dict
    else:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Incorrect Checkpoint path: {ckpt_path} (This should be an absolute path)")

        state_dict = torch.load(ckpt_path, map_location=device)
        return state_dict


def adapt_state_dict_to_fit_model_layer_names(model_state_dict: dict, source_ckpt: dict, exclude: list = [], solver: callable = None):
    """
    Given a model state dict and source checkpoints, the method tries to correct the keys in the model_state_dict to fit
    the ckpt in order to properly load the weights into the model. If unsuccessful - returns None
        :param model_state_dict:               the model state_dict
        :param source_ckpt:                         checkpoint dict
        :param exclude                  optional list for excluded layers
        :param solver:                  callable with signature (ckpt_key, ckpt_val, model_key, model_val)
                                        that returns a desired weight for ckpt_val.
        :return: renamed checkpoint dict (if possible)
    """
    if "net" in source_ckpt.keys():
        source_ckpt = source_ckpt["net"]
    model_state_dict_excluded = {k: v for k, v in model_state_dict.items() if not any(x in k for x in exclude)}
    new_ckpt_dict = {}
    for (ckpt_key, ckpt_val), (model_key, model_val) in zip(source_ckpt.items(), model_state_dict_excluded.items()):
        if solver is not None:
            ckpt_val = solver(ckpt_key, ckpt_val, model_key, model_val)
        if ckpt_val.shape != model_val.shape:
            raise ValueError(f"ckpt layer {ckpt_key} with shape {ckpt_val.shape} does not match {model_key}" f" with shape {model_val.shape} in the model")
        new_ckpt_dict[model_key] = ckpt_val
    return {"net": new_ckpt_dict}


def raise_informative_runtime_error(state_dict, checkpoint, exception_msg):
    """
    Given a model state dict and source checkpoints, the method calls "adapt_state_dict_to_fit_model_layer_names"
    and enhances the exception_msg if loading the checkpoint_dict via the conversion method is possible
    """
    try:
        new_ckpt_dict = adapt_state_dict_to_fit_model_layer_names(state_dict, checkpoint)
        temp_file = tempfile.NamedTemporaryFile().name + ".pt"
        torch.save(new_ckpt_dict, temp_file)
        exception_msg = (
            f"\n{'=' * 200}\n{str(exception_msg)} \nconvert ckpt via the utils.adapt_state_dict_to_fit_"
            f"model_layer_names method\na converted checkpoint file was saved in the path {temp_file}\n{'=' * 200}"
        )
    except ValueError as ex:  # IN CASE adapt_state_dict_to_fit_model_layer_names WAS UNSUCCESSFUL
        exception_msg = f"\n{'=' * 200} \nThe checkpoint and model shapes do no fit, e.g.: {ex}\n{'=' * 200}"
    finally:
        raise RuntimeError(exception_msg)


def load_checkpoint_to_model(
    net: torch.nn.Module,
    ckpt_local_path: str,
    load_backbone: bool = False,
    strict: Union[str, StrictLoad] = StrictLoad.NO_KEY_MATCHING,
    load_weights_only: bool = False,
    load_ema_as_net: bool = False,
    load_processing_params: bool = False,
):
    """
    Loads the state dict in ckpt_local_path to net and returns the checkpoint's state dict.


    :param load_ema_as_net: Will load the EMA inside the checkpoint file to the network when set
    :param ckpt_local_path: local path to the checkpoint file
    :param load_backbone: whether to load the checkpoint as a backbone
    :param net: network to load the checkpoint to
    :param strict:
    :param load_weights_only: Whether to ignore all other entries other then "net".
    :param load_processing_params: Whether to call set_dataset_processing_params on "processing_params" entry inside the
     checkpoint file (default=False).
    :return:
    """
    if isinstance(strict, str):
        strict = StrictLoad(strict)

    if load_backbone and not hasattr(net, "backbone"):
        raise ValueError("No backbone attribute in net - Can't load backbone weights")

    # LOAD THE LOCAL CHECKPOINT PATH INTO A state_dict OBJECT
    checkpoint = read_ckpt_state_dict(ckpt_path=ckpt_local_path)

    if load_ema_as_net:
        if "ema_net" not in checkpoint.keys():
            raise ValueError("Can't load ema network- no EMA network stored in checkpoint file")
        else:
            checkpoint["net"] = checkpoint["ema_net"]

    # LOAD THE CHECKPOINTS WEIGHTS TO THE MODEL
    if load_backbone:
        adaptive_load_state_dict(net.backbone, checkpoint, strict)
    else:
        adaptive_load_state_dict(net, checkpoint, strict)

    message_suffix = " checkpoint." if not load_ema_as_net else " EMA checkpoint."
    message_model = "model" if not load_backbone else "model's backbone"
    logger.info("Successfully loaded " + message_model + " weights from " + ckpt_local_path + message_suffix)

    if (isinstance(net, HasPredict) or (hasattr(net, "module") and isinstance(net.module, HasPredict))) and load_processing_params:
        if "processing_params" not in checkpoint.keys():
            raise ValueError("Can't load processing params - could not find any stored in checkpoint file.")
        try:
            net.set_dataset_processing_params(**checkpoint["processing_params"])
        except Exception as e:
            logger.warning(
                f"Could not set preprocessing pipeline from the checkpoint dataset: {e}. Before calling"
                "predict make sure to call set_dataset_processing_params."
            )

    if load_weights_only or load_backbone:
        # DISCARD ALL THE DATA STORED IN CHECKPOINT OTHER THAN THE WEIGHTS
        [checkpoint.pop(key) for key in list(checkpoint.keys()) if key != "net"]

    return checkpoint


class MissingPretrainedWeightsException(Exception):
    """Exception raised by unsupported pretrianed model.

    :param desc: explanation of the error
    """

    def __init__(self, desc):
        self.message = "Missing pretrained wights: " + desc
        super().__init__(self.message)


def _yolox_ckpt_solver(ckpt_key, ckpt_val, model_key, model_val):
    """
    Helper method for reshaping old pretrained checkpoint's focus weights to 6x6 conv weights.
    """

    if (
        ckpt_val.shape != model_val.shape
        and ckpt_key == "module._backbone._modules_list.0.conv.conv.weight"
        and model_key == "_backbone._modules_list.0.conv.weight"
    ):
        model_val.data[:, :, ::2, ::2] = ckpt_val.data[:, :3]
        model_val.data[:, :, 1::2, ::2] = ckpt_val.data[:, 3:6]
        model_val.data[:, :, ::2, 1::2] = ckpt_val.data[:, 6:9]
        model_val.data[:, :, 1::2, 1::2] = ckpt_val.data[:, 9:12]
        replacement = model_val
    else:
        replacement = ckpt_val

    return replacement


def load_pretrained_weights(model: torch.nn.Module, architecture: str, pretrained_weights: str):

    """
    Loads pretrained weights from the MODEL_URLS dictionary to model
    :param architecture: name of the model's architecture
    :param model: model to load pretrinaed weights for
    :param pretrained_weights: name for the pretrianed weights (i.e imagenet)
    :return: None
    """
    from super_gradients.common.object_names import Models

    model_url_key = architecture + "_" + str(pretrained_weights)
    if model_url_key not in MODEL_URLS.keys():
        raise MissingPretrainedWeightsException(model_url_key)

    url = MODEL_URLS[model_url_key]

    if architecture in {Models.YOLO_NAS_S, Models.YOLO_NAS_M, Models.YOLO_NAS_L}:
        logger.info(
            "License Notification: YOLO-NAS pre-trained weights are subjected to the specific license terms and conditions detailed in \n"
            "https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md\n"
            "By downloading the pre-trained weight files you agree to comply with these terms."
        )

    unique_filename = url.split("https://sghub.deci.ai/models/")[1].replace("/", "_").replace(" ", "_")
    map_location = torch.device("cpu")
    with wait_for_the_master(get_local_rank()):
        pretrained_state_dict = load_state_dict_from_url(url=url, map_location=map_location, file_name=unique_filename)
    _load_weights(architecture, model, pretrained_state_dict)


def _load_weights(architecture, model, pretrained_state_dict):
    if "ema_net" in pretrained_state_dict.keys():
        pretrained_state_dict["net"] = pretrained_state_dict["ema_net"]
    solver = _yolox_ckpt_solver if "yolox" in architecture else None
    adaptive_load_state_dict(net=model, state_dict=pretrained_state_dict, strict=StrictLoad.NO_KEY_MATCHING, solver=solver)


def load_pretrained_weights_local(model: torch.nn.Module, architecture: str, pretrained_weights: str):

    """
    Loads pretrained weights from the MODEL_URLS dictionary to model
    :param architecture: name of the model's architecture
    :param model: model to load pretrinaed weights for
    :param pretrained_weights: path tp pretrained weights
    :return: None
    """

    map_location = torch.device("cpu")

    pretrained_state_dict = torch.load(pretrained_weights, map_location=map_location)
    _load_weights(architecture, model, pretrained_state_dict)
