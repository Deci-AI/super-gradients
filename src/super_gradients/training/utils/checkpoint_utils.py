import collections
import os
import tempfile
from typing import Union, Mapping, Dict

import pkg_resources
import torch
from torch import nn, Tensor

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.data_interface.adnn_model_repository_data_interface import ADNNModelRepositoryDataInterfaces
from super_gradients.common.data_types import StrictLoad
from super_gradients.common.decorators.explicit_params_validator import explicit_params_validation
from super_gradients.module_interfaces import HasPredict
from super_gradients.training.pretrained_models import MODEL_URLS, DATASET_LICENSES
from super_gradients.training.utils.distributed_training_utils import wait_for_the_master
from super_gradients.common.environment.ddp_utils import get_local_rank
from super_gradients.training.utils.utils import unwrap_model
from super_gradients.common.factories.type_factory import TypeFactory
from super_gradients.common.decorators.factory_decorator import resolve_param

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

    transfered_weights = 0
    for name, value in model_state_dict.items():
        try:
            model.load_state_dict(collections.OrderedDict([(name, value)]), strict=False)
            transfered_weights += 1
        except RuntimeError:
            pass

    percentage_of_checkpoint = transfered_weights / len(model_state_dict)
    percentage_of_model = transfered_weights / len(model.state_dict())
    logger.debug(
        f"Transfered {transfered_weights} ({(100 * percentage_of_checkpoint):.2f}%) weights from the checkpoint. "
        f"{(100 * percentage_of_model):.2f}% of the model layers were initialized using checkpoint."
    )


def maybe_remove_module_prefix(state_dict: Mapping[str, Tensor], prefix: str = "module.") -> Mapping[str, Tensor]:
    """
    Checks is all the keys in `state_dict` start with `prefix` and if this is true removes this prefix.
    This function is intended to drop a "module." prefix from all keys in checkpoint that was saved
    with DataParallel/DistributedDataParallel wrapper.

    Since SG 3.1 we changed this behavior and always unwrap the model before saving the state_dict.
    However, to keep the compatibility with older checkpoints, we must do the 'cleanup' before loading the state_dict.

    :params: state_dict: The model state_dict
    :params: prefix: (str) prefix to remove. Default is "module."
    :return: state_dict: The model state_dict after removing the prefix

    """
    offset = len(prefix)
    if all([key.startswith(prefix) for key in state_dict.keys()]):
        state_dict = collections.OrderedDict([(key[offset:], value) for key, value in state_dict.items()])
    return state_dict


def adaptive_load_state_dict(net: torch.nn.Module, state_dict: dict, strict: Union[bool, StrictLoad], solver=None):
    """
    Adaptively loads state_dict to net, by adapting the state_dict to net's layer names first.
    :param net: (nn.Module) to load state_dict to
    :param state_dict: (dict) Checkpoint state_dict
    :param strict: (StrictLoad) key matching strictness
    :param solver: callable with signature (ckpt_key, ckpt_val, model_key, model_val)
                     that returns a desired weight for ckpt_val.
    :return:
    """
    state_dict = state_dict["net"] if "net" in state_dict else state_dict

    # This is a backward compatibility fix for checkpoints that were saved with DataParallel/DistributedDataParallel wrapper
    # and contains "module." prefix in all keys
    # If all keys start with "module.", then we remove it.
    state_dict = maybe_remove_module_prefix(state_dict)

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
    :param device: Target device where tensors should be loaded
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


class DefaultCheckpointSolver:
    """
    Implements the default behavior from adaptive_load_state_dict.
    If the model state dict and checkpoint state dict has no 1:1 matching by name,
    then default solver uses simple ordered matching.
    It assumes that order of layers in the checkpoint is the same as in the model and
    iterates over them simultaneously.
    If shape of the source and recipient tensors are different, solver raises an error.
    """

    def __call__(self, model_state_dict: Mapping[str, Tensor], checkpoint_state_dict: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """
        Map checkpoint state_dict to model state_dict.

        :param model_state_dict: (Mapping[str, Tensor]) A checkpoint state dict
        :param checkpoint_state_dict: (Mapping[str, Tensor]) A model state dict
        :return: (Mapping[str, Tensor]) New checkpoint state dict with keys/values converted to match model state_dict
        """
        new_ckpt_dict = {}
        for (ckpt_key, ckpt_val), (model_key, model_val) in zip(checkpoint_state_dict.items(), model_state_dict.items()):
            if ckpt_val.shape != model_val.shape:
                raise ValueError(f"ckpt layer {ckpt_key} with shape {ckpt_val.shape} does not match {model_key}" f" with shape {model_val.shape} in the model")
            new_ckpt_dict[model_key] = ckpt_val
        return new_ckpt_dict


class YoloXCheckpointSolver:
    """
    Implementation of checkpoint solver for old YoloX model checkpoints.
    """

    @classmethod
    def generate_mapping_table(cls) -> Mapping[str, str]:
        """
        Helper method to generate mapping table between olx YoloX checkpoints and the current YoloX layer names.
        :return: A mapping dictionary {checkpoint_key: model_key}
        """
        from super_gradients.common.object_names import Models
        from super_gradients.training import models

        all_mapping_keys = {}
        model_names = [Models.YOLOX_N, Models.YOLOX_T, Models.YOLOX_S, Models.YOLOX_M, Models.YOLOX_L]

        for model_name in model_names:
            model_url = MODEL_URLS[model_name + "_coco"]
            state_dict = load_state_dict_from_url(model_url, progress=True, map_location="cpu")

            model = models.get(model_name, num_classes=80)
            model_state_dict = model.state_dict()
            checkpoint_state_dict = maybe_remove_module_prefix(state_dict["net"])
            new_sd = {
                k: v
                for k, v in checkpoint_state_dict.items()
                if k not in {"stride", "_head.anchors._anchors", "_head.anchors._anchor_grid", "_head.anchors._stride", "_head._modules_list.14.stride"}
            }

            for (model_key, model_value), (checkpoint_key, checkpoint_value) in zip(model_state_dict.items(), new_sd.items()):
                if model_value.size() == checkpoint_value.size() and model_key.split(".")[-1] == checkpoint_key.split(".")[-1]:
                    if checkpoint_key in all_mapping_keys:
                        assert all_mapping_keys[checkpoint_key] == model_key
                    all_mapping_keys[checkpoint_key] = model_key
                else:
                    raise RuntimeError(
                        "Detected mismatch between model and checkpoint state dict keys."
                        f"Model key {model_key} of shape {model_value.size()} does not "
                        f"match checkpoint key {checkpoint_key} of shape {checkpoint_value.size()}"
                    )

        return all_mapping_keys

    def __init__(self):
        # The layers_rename_table below is a result of a manual mapping between the checkpoint keys and the model keys.
        # It was code-generated using YoloXCheckpointSolver.generate_mapping_table() method and tested for
        # correctness with:
        # tests.unit_tests.yolox_unit_test.TestYOLOX.test_yolo_x_checkpoint_solver.
        # tests.unit_tests.test_predict.TestModelPredict.test_detection_models

        self.layers_rename_table = {
            "_backbone._modules_list.0.conv.bn.bias": "_backbone._modules_list.0.bn.bias",
            "_backbone._modules_list.0.conv.bn.num_batches_tracked": "_backbone._modules_list.0.bn.num_batches_tracked",
            "_backbone._modules_list.0.conv.bn.running_mean": "_backbone._modules_list.0.bn.running_mean",
            "_backbone._modules_list.0.conv.bn.running_var": "_backbone._modules_list.0.bn.running_var",
            "_backbone._modules_list.0.conv.bn.weight": "_backbone._modules_list.0.bn.weight",
            "_backbone._modules_list.1.bn.bias": "_backbone._modules_list.1.bn.bias",
            "_backbone._modules_list.1.bn.num_batches_tracked": "_backbone._modules_list.1.bn.num_batches_tracked",
            "_backbone._modules_list.1.bn.running_mean": "_backbone._modules_list.1.bn.running_mean",
            "_backbone._modules_list.1.bn.running_var": "_backbone._modules_list.1.bn.running_var",
            "_backbone._modules_list.1.bn.weight": "_backbone._modules_list.1.bn.weight",
            "_backbone._modules_list.1.conv.bn.bias": "_backbone._modules_list.1.conv.bn.bias",
            "_backbone._modules_list.1.conv.bn.num_batches_tracked": "_backbone._modules_list.1.conv.bn.num_batches_tracked",
            "_backbone._modules_list.1.conv.bn.running_mean": "_backbone._modules_list.1.conv.bn.running_mean",
            "_backbone._modules_list.1.conv.bn.running_var": "_backbone._modules_list.1.conv.bn.running_var",
            "_backbone._modules_list.1.conv.bn.weight": "_backbone._modules_list.1.conv.bn.weight",
            "_backbone._modules_list.1.conv.conv.weight": "_backbone._modules_list.1.conv.conv.weight",
            "_backbone._modules_list.1.conv.weight": "_backbone._modules_list.1.conv.weight",
            "_backbone._modules_list.1.dconv.bn.bias": "_backbone._modules_list.1.dconv.bn.bias",
            "_backbone._modules_list.1.dconv.bn.num_batches_tracked": "_backbone._modules_list.1.dconv.bn.num_batches_tracked",
            "_backbone._modules_list.1.dconv.bn.running_mean": "_backbone._modules_list.1.dconv.bn.running_mean",
            "_backbone._modules_list.1.dconv.bn.running_var": "_backbone._modules_list.1.dconv.bn.running_var",
            "_backbone._modules_list.1.dconv.bn.weight": "_backbone._modules_list.1.dconv.bn.weight",
            "_backbone._modules_list.1.dconv.conv.weight": "_backbone._modules_list.1.dconv.conv.weight",
            "_backbone._modules_list.2.cv1.bn.bias": "_backbone._modules_list.2.conv1.bn.bias",
            "_backbone._modules_list.2.cv1.bn.num_batches_tracked": "_backbone._modules_list.2.conv1.bn.num_batches_tracked",
            "_backbone._modules_list.2.cv1.bn.running_mean": "_backbone._modules_list.2.conv1.bn.running_mean",
            "_backbone._modules_list.2.cv1.bn.running_var": "_backbone._modules_list.2.conv1.bn.running_var",
            "_backbone._modules_list.2.cv1.bn.weight": "_backbone._modules_list.2.conv1.bn.weight",
            "_backbone._modules_list.2.cv1.conv.weight": "_backbone._modules_list.2.conv1.conv.weight",
            "_backbone._modules_list.2.cv2.bn.bias": "_backbone._modules_list.2.conv2.bn.bias",
            "_backbone._modules_list.2.cv2.bn.num_batches_tracked": "_backbone._modules_list.2.conv2.bn.num_batches_tracked",
            "_backbone._modules_list.2.cv2.bn.running_mean": "_backbone._modules_list.2.conv2.bn.running_mean",
            "_backbone._modules_list.2.cv2.bn.running_var": "_backbone._modules_list.2.conv2.bn.running_var",
            "_backbone._modules_list.2.cv2.bn.weight": "_backbone._modules_list.2.conv2.bn.weight",
            "_backbone._modules_list.2.cv2.conv.weight": "_backbone._modules_list.2.conv2.conv.weight",
            "_backbone._modules_list.2.cv3.bn.bias": "_backbone._modules_list.2.conv3.bn.bias",
            "_backbone._modules_list.2.cv3.bn.num_batches_tracked": "_backbone._modules_list.2.conv3.bn.num_batches_tracked",
            "_backbone._modules_list.2.cv3.bn.running_mean": "_backbone._modules_list.2.conv3.bn.running_mean",
            "_backbone._modules_list.2.cv3.bn.running_var": "_backbone._modules_list.2.conv3.bn.running_var",
            "_backbone._modules_list.2.cv3.bn.weight": "_backbone._modules_list.2.conv3.bn.weight",
            "_backbone._modules_list.2.cv3.conv.weight": "_backbone._modules_list.2.conv3.conv.weight",
            "_backbone._modules_list.2.m.0.cv1.bn.bias": "_backbone._modules_list.2.bottlenecks.0.cv1.bn.bias",
            "_backbone._modules_list.2.m.0.cv1.bn.num_batches_tracked": "_backbone._modules_list.2.bottlenecks.0.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.2.m.0.cv1.bn.running_mean": "_backbone._modules_list.2.bottlenecks.0.cv1.bn.running_mean",
            "_backbone._modules_list.2.m.0.cv1.bn.running_var": "_backbone._modules_list.2.bottlenecks.0.cv1.bn.running_var",
            "_backbone._modules_list.2.m.0.cv1.bn.weight": "_backbone._modules_list.2.bottlenecks.0.cv1.bn.weight",
            "_backbone._modules_list.2.m.0.cv1.conv.weight": "_backbone._modules_list.2.bottlenecks.0.cv1.conv.weight",
            "_backbone._modules_list.2.m.0.cv2.bn.bias": "_backbone._modules_list.2.bottlenecks.0.cv2.bn.bias",
            "_backbone._modules_list.2.m.0.cv2.bn.num_batches_tracked": "_backbone._modules_list.2.bottlenecks.0.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.2.m.0.cv2.bn.running_mean": "_backbone._modules_list.2.bottlenecks.0.cv2.bn.running_mean",
            "_backbone._modules_list.2.m.0.cv2.bn.running_var": "_backbone._modules_list.2.bottlenecks.0.cv2.bn.running_var",
            "_backbone._modules_list.2.m.0.cv2.bn.weight": "_backbone._modules_list.2.bottlenecks.0.cv2.bn.weight",
            "_backbone._modules_list.2.m.0.cv2.conv.bn.bias": "_backbone._modules_list.2.bottlenecks.0.cv2.conv.bn.bias",
            "_backbone._modules_list.2.m.0.cv2.conv.bn.num_batches_tracked": "_backbone._modules_list.2.bottlenecks.0.cv2.conv.bn.num_batches_tracked",
            "_backbone._modules_list.2.m.0.cv2.conv.bn.running_mean": "_backbone._modules_list.2.bottlenecks.0.cv2.conv.bn.running_mean",
            "_backbone._modules_list.2.m.0.cv2.conv.bn.running_var": "_backbone._modules_list.2.bottlenecks.0.cv2.conv.bn.running_var",
            "_backbone._modules_list.2.m.0.cv2.conv.bn.weight": "_backbone._modules_list.2.bottlenecks.0.cv2.conv.bn.weight",
            "_backbone._modules_list.2.m.0.cv2.conv.conv.weight": "_backbone._modules_list.2.bottlenecks.0.cv2.conv.conv.weight",
            "_backbone._modules_list.2.m.0.cv2.conv.weight": "_backbone._modules_list.2.bottlenecks.0.cv2.conv.weight",
            "_backbone._modules_list.2.m.0.cv2.dconv.bn.bias": "_backbone._modules_list.2.bottlenecks.0.cv2.dconv.bn.bias",
            "_backbone._modules_list.2.m.0.cv2.dconv.bn.num_batches_tracked": "_backbone._modules_list.2.bottlenecks.0.cv2.dconv.bn.num_batches_tracked",
            "_backbone._modules_list.2.m.0.cv2.dconv.bn.running_mean": "_backbone._modules_list.2.bottlenecks.0.cv2.dconv.bn.running_mean",
            "_backbone._modules_list.2.m.0.cv2.dconv.bn.running_var": "_backbone._modules_list.2.bottlenecks.0.cv2.dconv.bn.running_var",
            "_backbone._modules_list.2.m.0.cv2.dconv.bn.weight": "_backbone._modules_list.2.bottlenecks.0.cv2.dconv.bn.weight",
            "_backbone._modules_list.2.m.0.cv2.dconv.conv.weight": "_backbone._modules_list.2.bottlenecks.0.cv2.dconv.conv.weight",
            "_backbone._modules_list.2.m.1.cv1.bn.bias": "_backbone._modules_list.2.bottlenecks.1.cv1.bn.bias",
            "_backbone._modules_list.2.m.1.cv1.bn.num_batches_tracked": "_backbone._modules_list.2.bottlenecks.1.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.2.m.1.cv1.bn.running_mean": "_backbone._modules_list.2.bottlenecks.1.cv1.bn.running_mean",
            "_backbone._modules_list.2.m.1.cv1.bn.running_var": "_backbone._modules_list.2.bottlenecks.1.cv1.bn.running_var",
            "_backbone._modules_list.2.m.1.cv1.bn.weight": "_backbone._modules_list.2.bottlenecks.1.cv1.bn.weight",
            "_backbone._modules_list.2.m.1.cv1.conv.weight": "_backbone._modules_list.2.bottlenecks.1.cv1.conv.weight",
            "_backbone._modules_list.2.m.1.cv2.bn.bias": "_backbone._modules_list.2.bottlenecks.1.cv2.bn.bias",
            "_backbone._modules_list.2.m.1.cv2.bn.num_batches_tracked": "_backbone._modules_list.2.bottlenecks.1.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.2.m.1.cv2.bn.running_mean": "_backbone._modules_list.2.bottlenecks.1.cv2.bn.running_mean",
            "_backbone._modules_list.2.m.1.cv2.bn.running_var": "_backbone._modules_list.2.bottlenecks.1.cv2.bn.running_var",
            "_backbone._modules_list.2.m.1.cv2.bn.weight": "_backbone._modules_list.2.bottlenecks.1.cv2.bn.weight",
            "_backbone._modules_list.2.m.1.cv2.conv.weight": "_backbone._modules_list.2.bottlenecks.1.cv2.conv.weight",
            "_backbone._modules_list.2.m.2.cv1.bn.bias": "_backbone._modules_list.2.bottlenecks.2.cv1.bn.bias",
            "_backbone._modules_list.2.m.2.cv1.bn.num_batches_tracked": "_backbone._modules_list.2.bottlenecks.2.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.2.m.2.cv1.bn.running_mean": "_backbone._modules_list.2.bottlenecks.2.cv1.bn.running_mean",
            "_backbone._modules_list.2.m.2.cv1.bn.running_var": "_backbone._modules_list.2.bottlenecks.2.cv1.bn.running_var",
            "_backbone._modules_list.2.m.2.cv1.bn.weight": "_backbone._modules_list.2.bottlenecks.2.cv1.bn.weight",
            "_backbone._modules_list.2.m.2.cv1.conv.weight": "_backbone._modules_list.2.bottlenecks.2.cv1.conv.weight",
            "_backbone._modules_list.2.m.2.cv2.bn.bias": "_backbone._modules_list.2.bottlenecks.2.cv2.bn.bias",
            "_backbone._modules_list.2.m.2.cv2.bn.num_batches_tracked": "_backbone._modules_list.2.bottlenecks.2.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.2.m.2.cv2.bn.running_mean": "_backbone._modules_list.2.bottlenecks.2.cv2.bn.running_mean",
            "_backbone._modules_list.2.m.2.cv2.bn.running_var": "_backbone._modules_list.2.bottlenecks.2.cv2.bn.running_var",
            "_backbone._modules_list.2.m.2.cv2.bn.weight": "_backbone._modules_list.2.bottlenecks.2.cv2.bn.weight",
            "_backbone._modules_list.2.m.2.cv2.conv.weight": "_backbone._modules_list.2.bottlenecks.2.cv2.conv.weight",
            "_backbone._modules_list.3.bn.bias": "_backbone._modules_list.3.bn.bias",
            "_backbone._modules_list.3.bn.num_batches_tracked": "_backbone._modules_list.3.bn.num_batches_tracked",
            "_backbone._modules_list.3.bn.running_mean": "_backbone._modules_list.3.bn.running_mean",
            "_backbone._modules_list.3.bn.running_var": "_backbone._modules_list.3.bn.running_var",
            "_backbone._modules_list.3.bn.weight": "_backbone._modules_list.3.bn.weight",
            "_backbone._modules_list.3.conv.bn.bias": "_backbone._modules_list.3.conv.bn.bias",
            "_backbone._modules_list.3.conv.bn.num_batches_tracked": "_backbone._modules_list.3.conv.bn.num_batches_tracked",
            "_backbone._modules_list.3.conv.bn.running_mean": "_backbone._modules_list.3.conv.bn.running_mean",
            "_backbone._modules_list.3.conv.bn.running_var": "_backbone._modules_list.3.conv.bn.running_var",
            "_backbone._modules_list.3.conv.bn.weight": "_backbone._modules_list.3.conv.bn.weight",
            "_backbone._modules_list.3.conv.conv.weight": "_backbone._modules_list.3.conv.conv.weight",
            "_backbone._modules_list.3.conv.weight": "_backbone._modules_list.3.conv.weight",
            "_backbone._modules_list.3.dconv.bn.bias": "_backbone._modules_list.3.dconv.bn.bias",
            "_backbone._modules_list.3.dconv.bn.num_batches_tracked": "_backbone._modules_list.3.dconv.bn.num_batches_tracked",
            "_backbone._modules_list.3.dconv.bn.running_mean": "_backbone._modules_list.3.dconv.bn.running_mean",
            "_backbone._modules_list.3.dconv.bn.running_var": "_backbone._modules_list.3.dconv.bn.running_var",
            "_backbone._modules_list.3.dconv.bn.weight": "_backbone._modules_list.3.dconv.bn.weight",
            "_backbone._modules_list.3.dconv.conv.weight": "_backbone._modules_list.3.dconv.conv.weight",
            "_backbone._modules_list.4.cv1.bn.bias": "_backbone._modules_list.4.conv1.bn.bias",
            "_backbone._modules_list.4.cv1.bn.num_batches_tracked": "_backbone._modules_list.4.conv1.bn.num_batches_tracked",
            "_backbone._modules_list.4.cv1.bn.running_mean": "_backbone._modules_list.4.conv1.bn.running_mean",
            "_backbone._modules_list.4.cv1.bn.running_var": "_backbone._modules_list.4.conv1.bn.running_var",
            "_backbone._modules_list.4.cv1.bn.weight": "_backbone._modules_list.4.conv1.bn.weight",
            "_backbone._modules_list.4.cv1.conv.weight": "_backbone._modules_list.4.conv1.conv.weight",
            "_backbone._modules_list.4.cv2.bn.bias": "_backbone._modules_list.4.conv2.bn.bias",
            "_backbone._modules_list.4.cv2.bn.num_batches_tracked": "_backbone._modules_list.4.conv2.bn.num_batches_tracked",
            "_backbone._modules_list.4.cv2.bn.running_mean": "_backbone._modules_list.4.conv2.bn.running_mean",
            "_backbone._modules_list.4.cv2.bn.running_var": "_backbone._modules_list.4.conv2.bn.running_var",
            "_backbone._modules_list.4.cv2.bn.weight": "_backbone._modules_list.4.conv2.bn.weight",
            "_backbone._modules_list.4.cv2.conv.weight": "_backbone._modules_list.4.conv2.conv.weight",
            "_backbone._modules_list.4.cv3.bn.bias": "_backbone._modules_list.4.conv3.bn.bias",
            "_backbone._modules_list.4.cv3.bn.num_batches_tracked": "_backbone._modules_list.4.conv3.bn.num_batches_tracked",
            "_backbone._modules_list.4.cv3.bn.running_mean": "_backbone._modules_list.4.conv3.bn.running_mean",
            "_backbone._modules_list.4.cv3.bn.running_var": "_backbone._modules_list.4.conv3.bn.running_var",
            "_backbone._modules_list.4.cv3.bn.weight": "_backbone._modules_list.4.conv3.bn.weight",
            "_backbone._modules_list.4.cv3.conv.weight": "_backbone._modules_list.4.conv3.conv.weight",
            "_backbone._modules_list.4.m.0.cv1.bn.bias": "_backbone._modules_list.4.bottlenecks.0.cv1.bn.bias",
            "_backbone._modules_list.4.m.0.cv1.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.0.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.0.cv1.bn.running_mean": "_backbone._modules_list.4.bottlenecks.0.cv1.bn.running_mean",
            "_backbone._modules_list.4.m.0.cv1.bn.running_var": "_backbone._modules_list.4.bottlenecks.0.cv1.bn.running_var",
            "_backbone._modules_list.4.m.0.cv1.bn.weight": "_backbone._modules_list.4.bottlenecks.0.cv1.bn.weight",
            "_backbone._modules_list.4.m.0.cv1.conv.weight": "_backbone._modules_list.4.bottlenecks.0.cv1.conv.weight",
            "_backbone._modules_list.4.m.0.cv2.bn.bias": "_backbone._modules_list.4.bottlenecks.0.cv2.bn.bias",
            "_backbone._modules_list.4.m.0.cv2.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.0.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.0.cv2.bn.running_mean": "_backbone._modules_list.4.bottlenecks.0.cv2.bn.running_mean",
            "_backbone._modules_list.4.m.0.cv2.bn.running_var": "_backbone._modules_list.4.bottlenecks.0.cv2.bn.running_var",
            "_backbone._modules_list.4.m.0.cv2.bn.weight": "_backbone._modules_list.4.bottlenecks.0.cv2.bn.weight",
            "_backbone._modules_list.4.m.0.cv2.conv.bn.bias": "_backbone._modules_list.4.bottlenecks.0.cv2.conv.bn.bias",
            "_backbone._modules_list.4.m.0.cv2.conv.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.0.cv2.conv.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.0.cv2.conv.bn.running_mean": "_backbone._modules_list.4.bottlenecks.0.cv2.conv.bn.running_mean",
            "_backbone._modules_list.4.m.0.cv2.conv.bn.running_var": "_backbone._modules_list.4.bottlenecks.0.cv2.conv.bn.running_var",
            "_backbone._modules_list.4.m.0.cv2.conv.bn.weight": "_backbone._modules_list.4.bottlenecks.0.cv2.conv.bn.weight",
            "_backbone._modules_list.4.m.0.cv2.conv.conv.weight": "_backbone._modules_list.4.bottlenecks.0.cv2.conv.conv.weight",
            "_backbone._modules_list.4.m.0.cv2.conv.weight": "_backbone._modules_list.4.bottlenecks.0.cv2.conv.weight",
            "_backbone._modules_list.4.m.0.cv2.dconv.bn.bias": "_backbone._modules_list.4.bottlenecks.0.cv2.dconv.bn.bias",
            "_backbone._modules_list.4.m.0.cv2.dconv.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.0.cv2.dconv.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.0.cv2.dconv.bn.running_mean": "_backbone._modules_list.4.bottlenecks.0.cv2.dconv.bn.running_mean",
            "_backbone._modules_list.4.m.0.cv2.dconv.bn.running_var": "_backbone._modules_list.4.bottlenecks.0.cv2.dconv.bn.running_var",
            "_backbone._modules_list.4.m.0.cv2.dconv.bn.weight": "_backbone._modules_list.4.bottlenecks.0.cv2.dconv.bn.weight",
            "_backbone._modules_list.4.m.0.cv2.dconv.conv.weight": "_backbone._modules_list.4.bottlenecks.0.cv2.dconv.conv.weight",
            "_backbone._modules_list.4.m.1.cv1.bn.bias": "_backbone._modules_list.4.bottlenecks.1.cv1.bn.bias",
            "_backbone._modules_list.4.m.1.cv1.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.1.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.1.cv1.bn.running_mean": "_backbone._modules_list.4.bottlenecks.1.cv1.bn.running_mean",
            "_backbone._modules_list.4.m.1.cv1.bn.running_var": "_backbone._modules_list.4.bottlenecks.1.cv1.bn.running_var",
            "_backbone._modules_list.4.m.1.cv1.bn.weight": "_backbone._modules_list.4.bottlenecks.1.cv1.bn.weight",
            "_backbone._modules_list.4.m.1.cv1.conv.weight": "_backbone._modules_list.4.bottlenecks.1.cv1.conv.weight",
            "_backbone._modules_list.4.m.1.cv2.bn.bias": "_backbone._modules_list.4.bottlenecks.1.cv2.bn.bias",
            "_backbone._modules_list.4.m.1.cv2.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.1.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.1.cv2.bn.running_mean": "_backbone._modules_list.4.bottlenecks.1.cv2.bn.running_mean",
            "_backbone._modules_list.4.m.1.cv2.bn.running_var": "_backbone._modules_list.4.bottlenecks.1.cv2.bn.running_var",
            "_backbone._modules_list.4.m.1.cv2.bn.weight": "_backbone._modules_list.4.bottlenecks.1.cv2.bn.weight",
            "_backbone._modules_list.4.m.1.cv2.conv.bn.bias": "_backbone._modules_list.4.bottlenecks.1.cv2.conv.bn.bias",
            "_backbone._modules_list.4.m.1.cv2.conv.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.1.cv2.conv.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.1.cv2.conv.bn.running_mean": "_backbone._modules_list.4.bottlenecks.1.cv2.conv.bn.running_mean",
            "_backbone._modules_list.4.m.1.cv2.conv.bn.running_var": "_backbone._modules_list.4.bottlenecks.1.cv2.conv.bn.running_var",
            "_backbone._modules_list.4.m.1.cv2.conv.bn.weight": "_backbone._modules_list.4.bottlenecks.1.cv2.conv.bn.weight",
            "_backbone._modules_list.4.m.1.cv2.conv.conv.weight": "_backbone._modules_list.4.bottlenecks.1.cv2.conv.conv.weight",
            "_backbone._modules_list.4.m.1.cv2.conv.weight": "_backbone._modules_list.4.bottlenecks.1.cv2.conv.weight",
            "_backbone._modules_list.4.m.1.cv2.dconv.bn.bias": "_backbone._modules_list.4.bottlenecks.1.cv2.dconv.bn.bias",
            "_backbone._modules_list.4.m.1.cv2.dconv.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.1.cv2.dconv.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.1.cv2.dconv.bn.running_mean": "_backbone._modules_list.4.bottlenecks.1.cv2.dconv.bn.running_mean",
            "_backbone._modules_list.4.m.1.cv2.dconv.bn.running_var": "_backbone._modules_list.4.bottlenecks.1.cv2.dconv.bn.running_var",
            "_backbone._modules_list.4.m.1.cv2.dconv.bn.weight": "_backbone._modules_list.4.bottlenecks.1.cv2.dconv.bn.weight",
            "_backbone._modules_list.4.m.1.cv2.dconv.conv.weight": "_backbone._modules_list.4.bottlenecks.1.cv2.dconv.conv.weight",
            "_backbone._modules_list.4.m.2.cv1.bn.bias": "_backbone._modules_list.4.bottlenecks.2.cv1.bn.bias",
            "_backbone._modules_list.4.m.2.cv1.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.2.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.2.cv1.bn.running_mean": "_backbone._modules_list.4.bottlenecks.2.cv1.bn.running_mean",
            "_backbone._modules_list.4.m.2.cv1.bn.running_var": "_backbone._modules_list.4.bottlenecks.2.cv1.bn.running_var",
            "_backbone._modules_list.4.m.2.cv1.bn.weight": "_backbone._modules_list.4.bottlenecks.2.cv1.bn.weight",
            "_backbone._modules_list.4.m.2.cv1.conv.weight": "_backbone._modules_list.4.bottlenecks.2.cv1.conv.weight",
            "_backbone._modules_list.4.m.2.cv2.bn.bias": "_backbone._modules_list.4.bottlenecks.2.cv2.bn.bias",
            "_backbone._modules_list.4.m.2.cv2.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.2.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.2.cv2.bn.running_mean": "_backbone._modules_list.4.bottlenecks.2.cv2.bn.running_mean",
            "_backbone._modules_list.4.m.2.cv2.bn.running_var": "_backbone._modules_list.4.bottlenecks.2.cv2.bn.running_var",
            "_backbone._modules_list.4.m.2.cv2.bn.weight": "_backbone._modules_list.4.bottlenecks.2.cv2.bn.weight",
            "_backbone._modules_list.4.m.2.cv2.conv.bn.bias": "_backbone._modules_list.4.bottlenecks.2.cv2.conv.bn.bias",
            "_backbone._modules_list.4.m.2.cv2.conv.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.2.cv2.conv.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.2.cv2.conv.bn.running_mean": "_backbone._modules_list.4.bottlenecks.2.cv2.conv.bn.running_mean",
            "_backbone._modules_list.4.m.2.cv2.conv.bn.running_var": "_backbone._modules_list.4.bottlenecks.2.cv2.conv.bn.running_var",
            "_backbone._modules_list.4.m.2.cv2.conv.bn.weight": "_backbone._modules_list.4.bottlenecks.2.cv2.conv.bn.weight",
            "_backbone._modules_list.4.m.2.cv2.conv.conv.weight": "_backbone._modules_list.4.bottlenecks.2.cv2.conv.conv.weight",
            "_backbone._modules_list.4.m.2.cv2.conv.weight": "_backbone._modules_list.4.bottlenecks.2.cv2.conv.weight",
            "_backbone._modules_list.4.m.2.cv2.dconv.bn.bias": "_backbone._modules_list.4.bottlenecks.2.cv2.dconv.bn.bias",
            "_backbone._modules_list.4.m.2.cv2.dconv.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.2.cv2.dconv.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.2.cv2.dconv.bn.running_mean": "_backbone._modules_list.4.bottlenecks.2.cv2.dconv.bn.running_mean",
            "_backbone._modules_list.4.m.2.cv2.dconv.bn.running_var": "_backbone._modules_list.4.bottlenecks.2.cv2.dconv.bn.running_var",
            "_backbone._modules_list.4.m.2.cv2.dconv.bn.weight": "_backbone._modules_list.4.bottlenecks.2.cv2.dconv.bn.weight",
            "_backbone._modules_list.4.m.2.cv2.dconv.conv.weight": "_backbone._modules_list.4.bottlenecks.2.cv2.dconv.conv.weight",
            "_backbone._modules_list.4.m.3.cv1.bn.bias": "_backbone._modules_list.4.bottlenecks.3.cv1.bn.bias",
            "_backbone._modules_list.4.m.3.cv1.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.3.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.3.cv1.bn.running_mean": "_backbone._modules_list.4.bottlenecks.3.cv1.bn.running_mean",
            "_backbone._modules_list.4.m.3.cv1.bn.running_var": "_backbone._modules_list.4.bottlenecks.3.cv1.bn.running_var",
            "_backbone._modules_list.4.m.3.cv1.bn.weight": "_backbone._modules_list.4.bottlenecks.3.cv1.bn.weight",
            "_backbone._modules_list.4.m.3.cv1.conv.weight": "_backbone._modules_list.4.bottlenecks.3.cv1.conv.weight",
            "_backbone._modules_list.4.m.3.cv2.bn.bias": "_backbone._modules_list.4.bottlenecks.3.cv2.bn.bias",
            "_backbone._modules_list.4.m.3.cv2.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.3.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.3.cv2.bn.running_mean": "_backbone._modules_list.4.bottlenecks.3.cv2.bn.running_mean",
            "_backbone._modules_list.4.m.3.cv2.bn.running_var": "_backbone._modules_list.4.bottlenecks.3.cv2.bn.running_var",
            "_backbone._modules_list.4.m.3.cv2.bn.weight": "_backbone._modules_list.4.bottlenecks.3.cv2.bn.weight",
            "_backbone._modules_list.4.m.3.cv2.conv.weight": "_backbone._modules_list.4.bottlenecks.3.cv2.conv.weight",
            "_backbone._modules_list.4.m.4.cv1.bn.bias": "_backbone._modules_list.4.bottlenecks.4.cv1.bn.bias",
            "_backbone._modules_list.4.m.4.cv1.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.4.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.4.cv1.bn.running_mean": "_backbone._modules_list.4.bottlenecks.4.cv1.bn.running_mean",
            "_backbone._modules_list.4.m.4.cv1.bn.running_var": "_backbone._modules_list.4.bottlenecks.4.cv1.bn.running_var",
            "_backbone._modules_list.4.m.4.cv1.bn.weight": "_backbone._modules_list.4.bottlenecks.4.cv1.bn.weight",
            "_backbone._modules_list.4.m.4.cv1.conv.weight": "_backbone._modules_list.4.bottlenecks.4.cv1.conv.weight",
            "_backbone._modules_list.4.m.4.cv2.bn.bias": "_backbone._modules_list.4.bottlenecks.4.cv2.bn.bias",
            "_backbone._modules_list.4.m.4.cv2.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.4.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.4.cv2.bn.running_mean": "_backbone._modules_list.4.bottlenecks.4.cv2.bn.running_mean",
            "_backbone._modules_list.4.m.4.cv2.bn.running_var": "_backbone._modules_list.4.bottlenecks.4.cv2.bn.running_var",
            "_backbone._modules_list.4.m.4.cv2.bn.weight": "_backbone._modules_list.4.bottlenecks.4.cv2.bn.weight",
            "_backbone._modules_list.4.m.4.cv2.conv.weight": "_backbone._modules_list.4.bottlenecks.4.cv2.conv.weight",
            "_backbone._modules_list.4.m.5.cv1.bn.bias": "_backbone._modules_list.4.bottlenecks.5.cv1.bn.bias",
            "_backbone._modules_list.4.m.5.cv1.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.5.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.5.cv1.bn.running_mean": "_backbone._modules_list.4.bottlenecks.5.cv1.bn.running_mean",
            "_backbone._modules_list.4.m.5.cv1.bn.running_var": "_backbone._modules_list.4.bottlenecks.5.cv1.bn.running_var",
            "_backbone._modules_list.4.m.5.cv1.bn.weight": "_backbone._modules_list.4.bottlenecks.5.cv1.bn.weight",
            "_backbone._modules_list.4.m.5.cv1.conv.weight": "_backbone._modules_list.4.bottlenecks.5.cv1.conv.weight",
            "_backbone._modules_list.4.m.5.cv2.bn.bias": "_backbone._modules_list.4.bottlenecks.5.cv2.bn.bias",
            "_backbone._modules_list.4.m.5.cv2.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.5.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.5.cv2.bn.running_mean": "_backbone._modules_list.4.bottlenecks.5.cv2.bn.running_mean",
            "_backbone._modules_list.4.m.5.cv2.bn.running_var": "_backbone._modules_list.4.bottlenecks.5.cv2.bn.running_var",
            "_backbone._modules_list.4.m.5.cv2.bn.weight": "_backbone._modules_list.4.bottlenecks.5.cv2.bn.weight",
            "_backbone._modules_list.4.m.5.cv2.conv.weight": "_backbone._modules_list.4.bottlenecks.5.cv2.conv.weight",
            "_backbone._modules_list.4.m.6.cv1.bn.bias": "_backbone._modules_list.4.bottlenecks.6.cv1.bn.bias",
            "_backbone._modules_list.4.m.6.cv1.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.6.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.6.cv1.bn.running_mean": "_backbone._modules_list.4.bottlenecks.6.cv1.bn.running_mean",
            "_backbone._modules_list.4.m.6.cv1.bn.running_var": "_backbone._modules_list.4.bottlenecks.6.cv1.bn.running_var",
            "_backbone._modules_list.4.m.6.cv1.bn.weight": "_backbone._modules_list.4.bottlenecks.6.cv1.bn.weight",
            "_backbone._modules_list.4.m.6.cv1.conv.weight": "_backbone._modules_list.4.bottlenecks.6.cv1.conv.weight",
            "_backbone._modules_list.4.m.6.cv2.bn.bias": "_backbone._modules_list.4.bottlenecks.6.cv2.bn.bias",
            "_backbone._modules_list.4.m.6.cv2.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.6.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.6.cv2.bn.running_mean": "_backbone._modules_list.4.bottlenecks.6.cv2.bn.running_mean",
            "_backbone._modules_list.4.m.6.cv2.bn.running_var": "_backbone._modules_list.4.bottlenecks.6.cv2.bn.running_var",
            "_backbone._modules_list.4.m.6.cv2.bn.weight": "_backbone._modules_list.4.bottlenecks.6.cv2.bn.weight",
            "_backbone._modules_list.4.m.6.cv2.conv.weight": "_backbone._modules_list.4.bottlenecks.6.cv2.conv.weight",
            "_backbone._modules_list.4.m.7.cv1.bn.bias": "_backbone._modules_list.4.bottlenecks.7.cv1.bn.bias",
            "_backbone._modules_list.4.m.7.cv1.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.7.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.7.cv1.bn.running_mean": "_backbone._modules_list.4.bottlenecks.7.cv1.bn.running_mean",
            "_backbone._modules_list.4.m.7.cv1.bn.running_var": "_backbone._modules_list.4.bottlenecks.7.cv1.bn.running_var",
            "_backbone._modules_list.4.m.7.cv1.bn.weight": "_backbone._modules_list.4.bottlenecks.7.cv1.bn.weight",
            "_backbone._modules_list.4.m.7.cv1.conv.weight": "_backbone._modules_list.4.bottlenecks.7.cv1.conv.weight",
            "_backbone._modules_list.4.m.7.cv2.bn.bias": "_backbone._modules_list.4.bottlenecks.7.cv2.bn.bias",
            "_backbone._modules_list.4.m.7.cv2.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.7.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.7.cv2.bn.running_mean": "_backbone._modules_list.4.bottlenecks.7.cv2.bn.running_mean",
            "_backbone._modules_list.4.m.7.cv2.bn.running_var": "_backbone._modules_list.4.bottlenecks.7.cv2.bn.running_var",
            "_backbone._modules_list.4.m.7.cv2.bn.weight": "_backbone._modules_list.4.bottlenecks.7.cv2.bn.weight",
            "_backbone._modules_list.4.m.7.cv2.conv.weight": "_backbone._modules_list.4.bottlenecks.7.cv2.conv.weight",
            "_backbone._modules_list.4.m.8.cv1.bn.bias": "_backbone._modules_list.4.bottlenecks.8.cv1.bn.bias",
            "_backbone._modules_list.4.m.8.cv1.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.8.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.8.cv1.bn.running_mean": "_backbone._modules_list.4.bottlenecks.8.cv1.bn.running_mean",
            "_backbone._modules_list.4.m.8.cv1.bn.running_var": "_backbone._modules_list.4.bottlenecks.8.cv1.bn.running_var",
            "_backbone._modules_list.4.m.8.cv1.bn.weight": "_backbone._modules_list.4.bottlenecks.8.cv1.bn.weight",
            "_backbone._modules_list.4.m.8.cv1.conv.weight": "_backbone._modules_list.4.bottlenecks.8.cv1.conv.weight",
            "_backbone._modules_list.4.m.8.cv2.bn.bias": "_backbone._modules_list.4.bottlenecks.8.cv2.bn.bias",
            "_backbone._modules_list.4.m.8.cv2.bn.num_batches_tracked": "_backbone._modules_list.4.bottlenecks.8.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.4.m.8.cv2.bn.running_mean": "_backbone._modules_list.4.bottlenecks.8.cv2.bn.running_mean",
            "_backbone._modules_list.4.m.8.cv2.bn.running_var": "_backbone._modules_list.4.bottlenecks.8.cv2.bn.running_var",
            "_backbone._modules_list.4.m.8.cv2.bn.weight": "_backbone._modules_list.4.bottlenecks.8.cv2.bn.weight",
            "_backbone._modules_list.4.m.8.cv2.conv.weight": "_backbone._modules_list.4.bottlenecks.8.cv2.conv.weight",
            "_backbone._modules_list.5.bn.bias": "_backbone._modules_list.5.bn.bias",
            "_backbone._modules_list.5.bn.num_batches_tracked": "_backbone._modules_list.5.bn.num_batches_tracked",
            "_backbone._modules_list.5.bn.running_mean": "_backbone._modules_list.5.bn.running_mean",
            "_backbone._modules_list.5.bn.running_var": "_backbone._modules_list.5.bn.running_var",
            "_backbone._modules_list.5.bn.weight": "_backbone._modules_list.5.bn.weight",
            "_backbone._modules_list.5.conv.bn.bias": "_backbone._modules_list.5.conv.bn.bias",
            "_backbone._modules_list.5.conv.bn.num_batches_tracked": "_backbone._modules_list.5.conv.bn.num_batches_tracked",
            "_backbone._modules_list.5.conv.bn.running_mean": "_backbone._modules_list.5.conv.bn.running_mean",
            "_backbone._modules_list.5.conv.bn.running_var": "_backbone._modules_list.5.conv.bn.running_var",
            "_backbone._modules_list.5.conv.bn.weight": "_backbone._modules_list.5.conv.bn.weight",
            "_backbone._modules_list.5.conv.conv.weight": "_backbone._modules_list.5.conv.conv.weight",
            "_backbone._modules_list.5.conv.weight": "_backbone._modules_list.5.conv.weight",
            "_backbone._modules_list.5.dconv.bn.bias": "_backbone._modules_list.5.dconv.bn.bias",
            "_backbone._modules_list.5.dconv.bn.num_batches_tracked": "_backbone._modules_list.5.dconv.bn.num_batches_tracked",
            "_backbone._modules_list.5.dconv.bn.running_mean": "_backbone._modules_list.5.dconv.bn.running_mean",
            "_backbone._modules_list.5.dconv.bn.running_var": "_backbone._modules_list.5.dconv.bn.running_var",
            "_backbone._modules_list.5.dconv.bn.weight": "_backbone._modules_list.5.dconv.bn.weight",
            "_backbone._modules_list.5.dconv.conv.weight": "_backbone._modules_list.5.dconv.conv.weight",
            "_backbone._modules_list.6.cv1.bn.bias": "_backbone._modules_list.6.conv1.bn.bias",
            "_backbone._modules_list.6.cv1.bn.num_batches_tracked": "_backbone._modules_list.6.conv1.bn.num_batches_tracked",
            "_backbone._modules_list.6.cv1.bn.running_mean": "_backbone._modules_list.6.conv1.bn.running_mean",
            "_backbone._modules_list.6.cv1.bn.running_var": "_backbone._modules_list.6.conv1.bn.running_var",
            "_backbone._modules_list.6.cv1.bn.weight": "_backbone._modules_list.6.conv1.bn.weight",
            "_backbone._modules_list.6.cv1.conv.weight": "_backbone._modules_list.6.conv1.conv.weight",
            "_backbone._modules_list.6.cv2.bn.bias": "_backbone._modules_list.6.conv2.bn.bias",
            "_backbone._modules_list.6.cv2.bn.num_batches_tracked": "_backbone._modules_list.6.conv2.bn.num_batches_tracked",
            "_backbone._modules_list.6.cv2.bn.running_mean": "_backbone._modules_list.6.conv2.bn.running_mean",
            "_backbone._modules_list.6.cv2.bn.running_var": "_backbone._modules_list.6.conv2.bn.running_var",
            "_backbone._modules_list.6.cv2.bn.weight": "_backbone._modules_list.6.conv2.bn.weight",
            "_backbone._modules_list.6.cv2.conv.weight": "_backbone._modules_list.6.conv2.conv.weight",
            "_backbone._modules_list.6.cv3.bn.bias": "_backbone._modules_list.6.conv3.bn.bias",
            "_backbone._modules_list.6.cv3.bn.num_batches_tracked": "_backbone._modules_list.6.conv3.bn.num_batches_tracked",
            "_backbone._modules_list.6.cv3.bn.running_mean": "_backbone._modules_list.6.conv3.bn.running_mean",
            "_backbone._modules_list.6.cv3.bn.running_var": "_backbone._modules_list.6.conv3.bn.running_var",
            "_backbone._modules_list.6.cv3.bn.weight": "_backbone._modules_list.6.conv3.bn.weight",
            "_backbone._modules_list.6.cv3.conv.weight": "_backbone._modules_list.6.conv3.conv.weight",
            "_backbone._modules_list.6.m.0.cv1.bn.bias": "_backbone._modules_list.6.bottlenecks.0.cv1.bn.bias",
            "_backbone._modules_list.6.m.0.cv1.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.0.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.0.cv1.bn.running_mean": "_backbone._modules_list.6.bottlenecks.0.cv1.bn.running_mean",
            "_backbone._modules_list.6.m.0.cv1.bn.running_var": "_backbone._modules_list.6.bottlenecks.0.cv1.bn.running_var",
            "_backbone._modules_list.6.m.0.cv1.bn.weight": "_backbone._modules_list.6.bottlenecks.0.cv1.bn.weight",
            "_backbone._modules_list.6.m.0.cv1.conv.weight": "_backbone._modules_list.6.bottlenecks.0.cv1.conv.weight",
            "_backbone._modules_list.6.m.0.cv2.bn.bias": "_backbone._modules_list.6.bottlenecks.0.cv2.bn.bias",
            "_backbone._modules_list.6.m.0.cv2.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.0.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.0.cv2.bn.running_mean": "_backbone._modules_list.6.bottlenecks.0.cv2.bn.running_mean",
            "_backbone._modules_list.6.m.0.cv2.bn.running_var": "_backbone._modules_list.6.bottlenecks.0.cv2.bn.running_var",
            "_backbone._modules_list.6.m.0.cv2.bn.weight": "_backbone._modules_list.6.bottlenecks.0.cv2.bn.weight",
            "_backbone._modules_list.6.m.0.cv2.conv.bn.bias": "_backbone._modules_list.6.bottlenecks.0.cv2.conv.bn.bias",
            "_backbone._modules_list.6.m.0.cv2.conv.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.0.cv2.conv.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.0.cv2.conv.bn.running_mean": "_backbone._modules_list.6.bottlenecks.0.cv2.conv.bn.running_mean",
            "_backbone._modules_list.6.m.0.cv2.conv.bn.running_var": "_backbone._modules_list.6.bottlenecks.0.cv2.conv.bn.running_var",
            "_backbone._modules_list.6.m.0.cv2.conv.bn.weight": "_backbone._modules_list.6.bottlenecks.0.cv2.conv.bn.weight",
            "_backbone._modules_list.6.m.0.cv2.conv.conv.weight": "_backbone._modules_list.6.bottlenecks.0.cv2.conv.conv.weight",
            "_backbone._modules_list.6.m.0.cv2.conv.weight": "_backbone._modules_list.6.bottlenecks.0.cv2.conv.weight",
            "_backbone._modules_list.6.m.0.cv2.dconv.bn.bias": "_backbone._modules_list.6.bottlenecks.0.cv2.dconv.bn.bias",
            "_backbone._modules_list.6.m.0.cv2.dconv.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.0.cv2.dconv.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.0.cv2.dconv.bn.running_mean": "_backbone._modules_list.6.bottlenecks.0.cv2.dconv.bn.running_mean",
            "_backbone._modules_list.6.m.0.cv2.dconv.bn.running_var": "_backbone._modules_list.6.bottlenecks.0.cv2.dconv.bn.running_var",
            "_backbone._modules_list.6.m.0.cv2.dconv.bn.weight": "_backbone._modules_list.6.bottlenecks.0.cv2.dconv.bn.weight",
            "_backbone._modules_list.6.m.0.cv2.dconv.conv.weight": "_backbone._modules_list.6.bottlenecks.0.cv2.dconv.conv.weight",
            "_backbone._modules_list.6.m.1.cv1.bn.bias": "_backbone._modules_list.6.bottlenecks.1.cv1.bn.bias",
            "_backbone._modules_list.6.m.1.cv1.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.1.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.1.cv1.bn.running_mean": "_backbone._modules_list.6.bottlenecks.1.cv1.bn.running_mean",
            "_backbone._modules_list.6.m.1.cv1.bn.running_var": "_backbone._modules_list.6.bottlenecks.1.cv1.bn.running_var",
            "_backbone._modules_list.6.m.1.cv1.bn.weight": "_backbone._modules_list.6.bottlenecks.1.cv1.bn.weight",
            "_backbone._modules_list.6.m.1.cv1.conv.weight": "_backbone._modules_list.6.bottlenecks.1.cv1.conv.weight",
            "_backbone._modules_list.6.m.1.cv2.bn.bias": "_backbone._modules_list.6.bottlenecks.1.cv2.bn.bias",
            "_backbone._modules_list.6.m.1.cv2.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.1.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.1.cv2.bn.running_mean": "_backbone._modules_list.6.bottlenecks.1.cv2.bn.running_mean",
            "_backbone._modules_list.6.m.1.cv2.bn.running_var": "_backbone._modules_list.6.bottlenecks.1.cv2.bn.running_var",
            "_backbone._modules_list.6.m.1.cv2.bn.weight": "_backbone._modules_list.6.bottlenecks.1.cv2.bn.weight",
            "_backbone._modules_list.6.m.1.cv2.conv.bn.bias": "_backbone._modules_list.6.bottlenecks.1.cv2.conv.bn.bias",
            "_backbone._modules_list.6.m.1.cv2.conv.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.1.cv2.conv.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.1.cv2.conv.bn.running_mean": "_backbone._modules_list.6.bottlenecks.1.cv2.conv.bn.running_mean",
            "_backbone._modules_list.6.m.1.cv2.conv.bn.running_var": "_backbone._modules_list.6.bottlenecks.1.cv2.conv.bn.running_var",
            "_backbone._modules_list.6.m.1.cv2.conv.bn.weight": "_backbone._modules_list.6.bottlenecks.1.cv2.conv.bn.weight",
            "_backbone._modules_list.6.m.1.cv2.conv.conv.weight": "_backbone._modules_list.6.bottlenecks.1.cv2.conv.conv.weight",
            "_backbone._modules_list.6.m.1.cv2.conv.weight": "_backbone._modules_list.6.bottlenecks.1.cv2.conv.weight",
            "_backbone._modules_list.6.m.1.cv2.dconv.bn.bias": "_backbone._modules_list.6.bottlenecks.1.cv2.dconv.bn.bias",
            "_backbone._modules_list.6.m.1.cv2.dconv.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.1.cv2.dconv.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.1.cv2.dconv.bn.running_mean": "_backbone._modules_list.6.bottlenecks.1.cv2.dconv.bn.running_mean",
            "_backbone._modules_list.6.m.1.cv2.dconv.bn.running_var": "_backbone._modules_list.6.bottlenecks.1.cv2.dconv.bn.running_var",
            "_backbone._modules_list.6.m.1.cv2.dconv.bn.weight": "_backbone._modules_list.6.bottlenecks.1.cv2.dconv.bn.weight",
            "_backbone._modules_list.6.m.1.cv2.dconv.conv.weight": "_backbone._modules_list.6.bottlenecks.1.cv2.dconv.conv.weight",
            "_backbone._modules_list.6.m.2.cv1.bn.bias": "_backbone._modules_list.6.bottlenecks.2.cv1.bn.bias",
            "_backbone._modules_list.6.m.2.cv1.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.2.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.2.cv1.bn.running_mean": "_backbone._modules_list.6.bottlenecks.2.cv1.bn.running_mean",
            "_backbone._modules_list.6.m.2.cv1.bn.running_var": "_backbone._modules_list.6.bottlenecks.2.cv1.bn.running_var",
            "_backbone._modules_list.6.m.2.cv1.bn.weight": "_backbone._modules_list.6.bottlenecks.2.cv1.bn.weight",
            "_backbone._modules_list.6.m.2.cv1.conv.weight": "_backbone._modules_list.6.bottlenecks.2.cv1.conv.weight",
            "_backbone._modules_list.6.m.2.cv2.bn.bias": "_backbone._modules_list.6.bottlenecks.2.cv2.bn.bias",
            "_backbone._modules_list.6.m.2.cv2.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.2.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.2.cv2.bn.running_mean": "_backbone._modules_list.6.bottlenecks.2.cv2.bn.running_mean",
            "_backbone._modules_list.6.m.2.cv2.bn.running_var": "_backbone._modules_list.6.bottlenecks.2.cv2.bn.running_var",
            "_backbone._modules_list.6.m.2.cv2.bn.weight": "_backbone._modules_list.6.bottlenecks.2.cv2.bn.weight",
            "_backbone._modules_list.6.m.2.cv2.conv.bn.bias": "_backbone._modules_list.6.bottlenecks.2.cv2.conv.bn.bias",
            "_backbone._modules_list.6.m.2.cv2.conv.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.2.cv2.conv.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.2.cv2.conv.bn.running_mean": "_backbone._modules_list.6.bottlenecks.2.cv2.conv.bn.running_mean",
            "_backbone._modules_list.6.m.2.cv2.conv.bn.running_var": "_backbone._modules_list.6.bottlenecks.2.cv2.conv.bn.running_var",
            "_backbone._modules_list.6.m.2.cv2.conv.bn.weight": "_backbone._modules_list.6.bottlenecks.2.cv2.conv.bn.weight",
            "_backbone._modules_list.6.m.2.cv2.conv.conv.weight": "_backbone._modules_list.6.bottlenecks.2.cv2.conv.conv.weight",
            "_backbone._modules_list.6.m.2.cv2.conv.weight": "_backbone._modules_list.6.bottlenecks.2.cv2.conv.weight",
            "_backbone._modules_list.6.m.2.cv2.dconv.bn.bias": "_backbone._modules_list.6.bottlenecks.2.cv2.dconv.bn.bias",
            "_backbone._modules_list.6.m.2.cv2.dconv.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.2.cv2.dconv.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.2.cv2.dconv.bn.running_mean": "_backbone._modules_list.6.bottlenecks.2.cv2.dconv.bn.running_mean",
            "_backbone._modules_list.6.m.2.cv2.dconv.bn.running_var": "_backbone._modules_list.6.bottlenecks.2.cv2.dconv.bn.running_var",
            "_backbone._modules_list.6.m.2.cv2.dconv.bn.weight": "_backbone._modules_list.6.bottlenecks.2.cv2.dconv.bn.weight",
            "_backbone._modules_list.6.m.2.cv2.dconv.conv.weight": "_backbone._modules_list.6.bottlenecks.2.cv2.dconv.conv.weight",
            "_backbone._modules_list.6.m.3.cv1.bn.bias": "_backbone._modules_list.6.bottlenecks.3.cv1.bn.bias",
            "_backbone._modules_list.6.m.3.cv1.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.3.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.3.cv1.bn.running_mean": "_backbone._modules_list.6.bottlenecks.3.cv1.bn.running_mean",
            "_backbone._modules_list.6.m.3.cv1.bn.running_var": "_backbone._modules_list.6.bottlenecks.3.cv1.bn.running_var",
            "_backbone._modules_list.6.m.3.cv1.bn.weight": "_backbone._modules_list.6.bottlenecks.3.cv1.bn.weight",
            "_backbone._modules_list.6.m.3.cv1.conv.weight": "_backbone._modules_list.6.bottlenecks.3.cv1.conv.weight",
            "_backbone._modules_list.6.m.3.cv2.bn.bias": "_backbone._modules_list.6.bottlenecks.3.cv2.bn.bias",
            "_backbone._modules_list.6.m.3.cv2.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.3.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.3.cv2.bn.running_mean": "_backbone._modules_list.6.bottlenecks.3.cv2.bn.running_mean",
            "_backbone._modules_list.6.m.3.cv2.bn.running_var": "_backbone._modules_list.6.bottlenecks.3.cv2.bn.running_var",
            "_backbone._modules_list.6.m.3.cv2.bn.weight": "_backbone._modules_list.6.bottlenecks.3.cv2.bn.weight",
            "_backbone._modules_list.6.m.3.cv2.conv.weight": "_backbone._modules_list.6.bottlenecks.3.cv2.conv.weight",
            "_backbone._modules_list.6.m.4.cv1.bn.bias": "_backbone._modules_list.6.bottlenecks.4.cv1.bn.bias",
            "_backbone._modules_list.6.m.4.cv1.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.4.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.4.cv1.bn.running_mean": "_backbone._modules_list.6.bottlenecks.4.cv1.bn.running_mean",
            "_backbone._modules_list.6.m.4.cv1.bn.running_var": "_backbone._modules_list.6.bottlenecks.4.cv1.bn.running_var",
            "_backbone._modules_list.6.m.4.cv1.bn.weight": "_backbone._modules_list.6.bottlenecks.4.cv1.bn.weight",
            "_backbone._modules_list.6.m.4.cv1.conv.weight": "_backbone._modules_list.6.bottlenecks.4.cv1.conv.weight",
            "_backbone._modules_list.6.m.4.cv2.bn.bias": "_backbone._modules_list.6.bottlenecks.4.cv2.bn.bias",
            "_backbone._modules_list.6.m.4.cv2.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.4.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.4.cv2.bn.running_mean": "_backbone._modules_list.6.bottlenecks.4.cv2.bn.running_mean",
            "_backbone._modules_list.6.m.4.cv2.bn.running_var": "_backbone._modules_list.6.bottlenecks.4.cv2.bn.running_var",
            "_backbone._modules_list.6.m.4.cv2.bn.weight": "_backbone._modules_list.6.bottlenecks.4.cv2.bn.weight",
            "_backbone._modules_list.6.m.4.cv2.conv.weight": "_backbone._modules_list.6.bottlenecks.4.cv2.conv.weight",
            "_backbone._modules_list.6.m.5.cv1.bn.bias": "_backbone._modules_list.6.bottlenecks.5.cv1.bn.bias",
            "_backbone._modules_list.6.m.5.cv1.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.5.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.5.cv1.bn.running_mean": "_backbone._modules_list.6.bottlenecks.5.cv1.bn.running_mean",
            "_backbone._modules_list.6.m.5.cv1.bn.running_var": "_backbone._modules_list.6.bottlenecks.5.cv1.bn.running_var",
            "_backbone._modules_list.6.m.5.cv1.bn.weight": "_backbone._modules_list.6.bottlenecks.5.cv1.bn.weight",
            "_backbone._modules_list.6.m.5.cv1.conv.weight": "_backbone._modules_list.6.bottlenecks.5.cv1.conv.weight",
            "_backbone._modules_list.6.m.5.cv2.bn.bias": "_backbone._modules_list.6.bottlenecks.5.cv2.bn.bias",
            "_backbone._modules_list.6.m.5.cv2.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.5.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.5.cv2.bn.running_mean": "_backbone._modules_list.6.bottlenecks.5.cv2.bn.running_mean",
            "_backbone._modules_list.6.m.5.cv2.bn.running_var": "_backbone._modules_list.6.bottlenecks.5.cv2.bn.running_var",
            "_backbone._modules_list.6.m.5.cv2.bn.weight": "_backbone._modules_list.6.bottlenecks.5.cv2.bn.weight",
            "_backbone._modules_list.6.m.5.cv2.conv.weight": "_backbone._modules_list.6.bottlenecks.5.cv2.conv.weight",
            "_backbone._modules_list.6.m.6.cv1.bn.bias": "_backbone._modules_list.6.bottlenecks.6.cv1.bn.bias",
            "_backbone._modules_list.6.m.6.cv1.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.6.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.6.cv1.bn.running_mean": "_backbone._modules_list.6.bottlenecks.6.cv1.bn.running_mean",
            "_backbone._modules_list.6.m.6.cv1.bn.running_var": "_backbone._modules_list.6.bottlenecks.6.cv1.bn.running_var",
            "_backbone._modules_list.6.m.6.cv1.bn.weight": "_backbone._modules_list.6.bottlenecks.6.cv1.bn.weight",
            "_backbone._modules_list.6.m.6.cv1.conv.weight": "_backbone._modules_list.6.bottlenecks.6.cv1.conv.weight",
            "_backbone._modules_list.6.m.6.cv2.bn.bias": "_backbone._modules_list.6.bottlenecks.6.cv2.bn.bias",
            "_backbone._modules_list.6.m.6.cv2.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.6.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.6.cv2.bn.running_mean": "_backbone._modules_list.6.bottlenecks.6.cv2.bn.running_mean",
            "_backbone._modules_list.6.m.6.cv2.bn.running_var": "_backbone._modules_list.6.bottlenecks.6.cv2.bn.running_var",
            "_backbone._modules_list.6.m.6.cv2.bn.weight": "_backbone._modules_list.6.bottlenecks.6.cv2.bn.weight",
            "_backbone._modules_list.6.m.6.cv2.conv.weight": "_backbone._modules_list.6.bottlenecks.6.cv2.conv.weight",
            "_backbone._modules_list.6.m.7.cv1.bn.bias": "_backbone._modules_list.6.bottlenecks.7.cv1.bn.bias",
            "_backbone._modules_list.6.m.7.cv1.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.7.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.7.cv1.bn.running_mean": "_backbone._modules_list.6.bottlenecks.7.cv1.bn.running_mean",
            "_backbone._modules_list.6.m.7.cv1.bn.running_var": "_backbone._modules_list.6.bottlenecks.7.cv1.bn.running_var",
            "_backbone._modules_list.6.m.7.cv1.bn.weight": "_backbone._modules_list.6.bottlenecks.7.cv1.bn.weight",
            "_backbone._modules_list.6.m.7.cv1.conv.weight": "_backbone._modules_list.6.bottlenecks.7.cv1.conv.weight",
            "_backbone._modules_list.6.m.7.cv2.bn.bias": "_backbone._modules_list.6.bottlenecks.7.cv2.bn.bias",
            "_backbone._modules_list.6.m.7.cv2.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.7.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.7.cv2.bn.running_mean": "_backbone._modules_list.6.bottlenecks.7.cv2.bn.running_mean",
            "_backbone._modules_list.6.m.7.cv2.bn.running_var": "_backbone._modules_list.6.bottlenecks.7.cv2.bn.running_var",
            "_backbone._modules_list.6.m.7.cv2.bn.weight": "_backbone._modules_list.6.bottlenecks.7.cv2.bn.weight",
            "_backbone._modules_list.6.m.7.cv2.conv.weight": "_backbone._modules_list.6.bottlenecks.7.cv2.conv.weight",
            "_backbone._modules_list.6.m.8.cv1.bn.bias": "_backbone._modules_list.6.bottlenecks.8.cv1.bn.bias",
            "_backbone._modules_list.6.m.8.cv1.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.8.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.8.cv1.bn.running_mean": "_backbone._modules_list.6.bottlenecks.8.cv1.bn.running_mean",
            "_backbone._modules_list.6.m.8.cv1.bn.running_var": "_backbone._modules_list.6.bottlenecks.8.cv1.bn.running_var",
            "_backbone._modules_list.6.m.8.cv1.bn.weight": "_backbone._modules_list.6.bottlenecks.8.cv1.bn.weight",
            "_backbone._modules_list.6.m.8.cv1.conv.weight": "_backbone._modules_list.6.bottlenecks.8.cv1.conv.weight",
            "_backbone._modules_list.6.m.8.cv2.bn.bias": "_backbone._modules_list.6.bottlenecks.8.cv2.bn.bias",
            "_backbone._modules_list.6.m.8.cv2.bn.num_batches_tracked": "_backbone._modules_list.6.bottlenecks.8.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.6.m.8.cv2.bn.running_mean": "_backbone._modules_list.6.bottlenecks.8.cv2.bn.running_mean",
            "_backbone._modules_list.6.m.8.cv2.bn.running_var": "_backbone._modules_list.6.bottlenecks.8.cv2.bn.running_var",
            "_backbone._modules_list.6.m.8.cv2.bn.weight": "_backbone._modules_list.6.bottlenecks.8.cv2.bn.weight",
            "_backbone._modules_list.6.m.8.cv2.conv.weight": "_backbone._modules_list.6.bottlenecks.8.cv2.conv.weight",
            "_backbone._modules_list.7.bn.bias": "_backbone._modules_list.7.bn.bias",
            "_backbone._modules_list.7.bn.num_batches_tracked": "_backbone._modules_list.7.bn.num_batches_tracked",
            "_backbone._modules_list.7.bn.running_mean": "_backbone._modules_list.7.bn.running_mean",
            "_backbone._modules_list.7.bn.running_var": "_backbone._modules_list.7.bn.running_var",
            "_backbone._modules_list.7.bn.weight": "_backbone._modules_list.7.bn.weight",
            "_backbone._modules_list.7.conv.bn.bias": "_backbone._modules_list.7.conv.bn.bias",
            "_backbone._modules_list.7.conv.bn.num_batches_tracked": "_backbone._modules_list.7.conv.bn.num_batches_tracked",
            "_backbone._modules_list.7.conv.bn.running_mean": "_backbone._modules_list.7.conv.bn.running_mean",
            "_backbone._modules_list.7.conv.bn.running_var": "_backbone._modules_list.7.conv.bn.running_var",
            "_backbone._modules_list.7.conv.bn.weight": "_backbone._modules_list.7.conv.bn.weight",
            "_backbone._modules_list.7.conv.conv.weight": "_backbone._modules_list.7.conv.conv.weight",
            "_backbone._modules_list.7.conv.weight": "_backbone._modules_list.7.conv.weight",
            "_backbone._modules_list.7.dconv.bn.bias": "_backbone._modules_list.7.dconv.bn.bias",
            "_backbone._modules_list.7.dconv.bn.num_batches_tracked": "_backbone._modules_list.7.dconv.bn.num_batches_tracked",
            "_backbone._modules_list.7.dconv.bn.running_mean": "_backbone._modules_list.7.dconv.bn.running_mean",
            "_backbone._modules_list.7.dconv.bn.running_var": "_backbone._modules_list.7.dconv.bn.running_var",
            "_backbone._modules_list.7.dconv.bn.weight": "_backbone._modules_list.7.dconv.bn.weight",
            "_backbone._modules_list.7.dconv.conv.weight": "_backbone._modules_list.7.dconv.conv.weight",
            "_backbone._modules_list.8.cv1.bn.bias": "_backbone._modules_list.8.cv1.bn.bias",
            "_backbone._modules_list.8.cv1.bn.num_batches_tracked": "_backbone._modules_list.8.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.8.cv1.bn.running_mean": "_backbone._modules_list.8.cv1.bn.running_mean",
            "_backbone._modules_list.8.cv1.bn.running_var": "_backbone._modules_list.8.cv1.bn.running_var",
            "_backbone._modules_list.8.cv1.bn.weight": "_backbone._modules_list.8.cv1.bn.weight",
            "_backbone._modules_list.8.cv1.conv.weight": "_backbone._modules_list.8.cv1.conv.weight",
            "_backbone._modules_list.8.cv2.bn.bias": "_backbone._modules_list.8.cv2.bn.bias",
            "_backbone._modules_list.8.cv2.bn.num_batches_tracked": "_backbone._modules_list.8.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.8.cv2.bn.running_mean": "_backbone._modules_list.8.cv2.bn.running_mean",
            "_backbone._modules_list.8.cv2.bn.running_var": "_backbone._modules_list.8.cv2.bn.running_var",
            "_backbone._modules_list.8.cv2.bn.weight": "_backbone._modules_list.8.cv2.bn.weight",
            "_backbone._modules_list.8.cv2.conv.weight": "_backbone._modules_list.8.cv2.conv.weight",
            "_backbone._modules_list.9.cv1.bn.bias": "_backbone._modules_list.9.conv1.bn.bias",
            "_backbone._modules_list.9.cv1.bn.num_batches_tracked": "_backbone._modules_list.9.conv1.bn.num_batches_tracked",
            "_backbone._modules_list.9.cv1.bn.running_mean": "_backbone._modules_list.9.conv1.bn.running_mean",
            "_backbone._modules_list.9.cv1.bn.running_var": "_backbone._modules_list.9.conv1.bn.running_var",
            "_backbone._modules_list.9.cv1.bn.weight": "_backbone._modules_list.9.conv1.bn.weight",
            "_backbone._modules_list.9.cv1.conv.weight": "_backbone._modules_list.9.conv1.conv.weight",
            "_backbone._modules_list.9.cv2.bn.bias": "_backbone._modules_list.9.conv2.bn.bias",
            "_backbone._modules_list.9.cv2.bn.num_batches_tracked": "_backbone._modules_list.9.conv2.bn.num_batches_tracked",
            "_backbone._modules_list.9.cv2.bn.running_mean": "_backbone._modules_list.9.conv2.bn.running_mean",
            "_backbone._modules_list.9.cv2.bn.running_var": "_backbone._modules_list.9.conv2.bn.running_var",
            "_backbone._modules_list.9.cv2.bn.weight": "_backbone._modules_list.9.conv2.bn.weight",
            "_backbone._modules_list.9.cv2.conv.weight": "_backbone._modules_list.9.conv2.conv.weight",
            "_backbone._modules_list.9.cv3.bn.bias": "_backbone._modules_list.9.conv3.bn.bias",
            "_backbone._modules_list.9.cv3.bn.num_batches_tracked": "_backbone._modules_list.9.conv3.bn.num_batches_tracked",
            "_backbone._modules_list.9.cv3.bn.running_mean": "_backbone._modules_list.9.conv3.bn.running_mean",
            "_backbone._modules_list.9.cv3.bn.running_var": "_backbone._modules_list.9.conv3.bn.running_var",
            "_backbone._modules_list.9.cv3.bn.weight": "_backbone._modules_list.9.conv3.bn.weight",
            "_backbone._modules_list.9.cv3.conv.weight": "_backbone._modules_list.9.conv3.conv.weight",
            "_backbone._modules_list.9.m.0.cv1.bn.bias": "_backbone._modules_list.9.bottlenecks.0.cv1.bn.bias",
            "_backbone._modules_list.9.m.0.cv1.bn.num_batches_tracked": "_backbone._modules_list.9.bottlenecks.0.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.9.m.0.cv1.bn.running_mean": "_backbone._modules_list.9.bottlenecks.0.cv1.bn.running_mean",
            "_backbone._modules_list.9.m.0.cv1.bn.running_var": "_backbone._modules_list.9.bottlenecks.0.cv1.bn.running_var",
            "_backbone._modules_list.9.m.0.cv1.bn.weight": "_backbone._modules_list.9.bottlenecks.0.cv1.bn.weight",
            "_backbone._modules_list.9.m.0.cv1.conv.weight": "_backbone._modules_list.9.bottlenecks.0.cv1.conv.weight",
            "_backbone._modules_list.9.m.0.cv2.bn.bias": "_backbone._modules_list.9.bottlenecks.0.cv2.bn.bias",
            "_backbone._modules_list.9.m.0.cv2.bn.num_batches_tracked": "_backbone._modules_list.9.bottlenecks.0.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.9.m.0.cv2.bn.running_mean": "_backbone._modules_list.9.bottlenecks.0.cv2.bn.running_mean",
            "_backbone._modules_list.9.m.0.cv2.bn.running_var": "_backbone._modules_list.9.bottlenecks.0.cv2.bn.running_var",
            "_backbone._modules_list.9.m.0.cv2.bn.weight": "_backbone._modules_list.9.bottlenecks.0.cv2.bn.weight",
            "_backbone._modules_list.9.m.0.cv2.conv.bn.bias": "_backbone._modules_list.9.bottlenecks.0.cv2.conv.bn.bias",
            "_backbone._modules_list.9.m.0.cv2.conv.bn.num_batches_tracked": "_backbone._modules_list.9.bottlenecks.0.cv2.conv.bn.num_batches_tracked",
            "_backbone._modules_list.9.m.0.cv2.conv.bn.running_mean": "_backbone._modules_list.9.bottlenecks.0.cv2.conv.bn.running_mean",
            "_backbone._modules_list.9.m.0.cv2.conv.bn.running_var": "_backbone._modules_list.9.bottlenecks.0.cv2.conv.bn.running_var",
            "_backbone._modules_list.9.m.0.cv2.conv.bn.weight": "_backbone._modules_list.9.bottlenecks.0.cv2.conv.bn.weight",
            "_backbone._modules_list.9.m.0.cv2.conv.conv.weight": "_backbone._modules_list.9.bottlenecks.0.cv2.conv.conv.weight",
            "_backbone._modules_list.9.m.0.cv2.conv.weight": "_backbone._modules_list.9.bottlenecks.0.cv2.conv.weight",
            "_backbone._modules_list.9.m.0.cv2.dconv.bn.bias": "_backbone._modules_list.9.bottlenecks.0.cv2.dconv.bn.bias",
            "_backbone._modules_list.9.m.0.cv2.dconv.bn.num_batches_tracked": "_backbone._modules_list.9.bottlenecks.0.cv2.dconv.bn.num_batches_tracked",
            "_backbone._modules_list.9.m.0.cv2.dconv.bn.running_mean": "_backbone._modules_list.9.bottlenecks.0.cv2.dconv.bn.running_mean",
            "_backbone._modules_list.9.m.0.cv2.dconv.bn.running_var": "_backbone._modules_list.9.bottlenecks.0.cv2.dconv.bn.running_var",
            "_backbone._modules_list.9.m.0.cv2.dconv.bn.weight": "_backbone._modules_list.9.bottlenecks.0.cv2.dconv.bn.weight",
            "_backbone._modules_list.9.m.0.cv2.dconv.conv.weight": "_backbone._modules_list.9.bottlenecks.0.cv2.dconv.conv.weight",
            "_backbone._modules_list.9.m.1.cv1.bn.bias": "_backbone._modules_list.9.bottlenecks.1.cv1.bn.bias",
            "_backbone._modules_list.9.m.1.cv1.bn.num_batches_tracked": "_backbone._modules_list.9.bottlenecks.1.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.9.m.1.cv1.bn.running_mean": "_backbone._modules_list.9.bottlenecks.1.cv1.bn.running_mean",
            "_backbone._modules_list.9.m.1.cv1.bn.running_var": "_backbone._modules_list.9.bottlenecks.1.cv1.bn.running_var",
            "_backbone._modules_list.9.m.1.cv1.bn.weight": "_backbone._modules_list.9.bottlenecks.1.cv1.bn.weight",
            "_backbone._modules_list.9.m.1.cv1.conv.weight": "_backbone._modules_list.9.bottlenecks.1.cv1.conv.weight",
            "_backbone._modules_list.9.m.1.cv2.bn.bias": "_backbone._modules_list.9.bottlenecks.1.cv2.bn.bias",
            "_backbone._modules_list.9.m.1.cv2.bn.num_batches_tracked": "_backbone._modules_list.9.bottlenecks.1.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.9.m.1.cv2.bn.running_mean": "_backbone._modules_list.9.bottlenecks.1.cv2.bn.running_mean",
            "_backbone._modules_list.9.m.1.cv2.bn.running_var": "_backbone._modules_list.9.bottlenecks.1.cv2.bn.running_var",
            "_backbone._modules_list.9.m.1.cv2.bn.weight": "_backbone._modules_list.9.bottlenecks.1.cv2.bn.weight",
            "_backbone._modules_list.9.m.1.cv2.conv.weight": "_backbone._modules_list.9.bottlenecks.1.cv2.conv.weight",
            "_backbone._modules_list.9.m.2.cv1.bn.bias": "_backbone._modules_list.9.bottlenecks.2.cv1.bn.bias",
            "_backbone._modules_list.9.m.2.cv1.bn.num_batches_tracked": "_backbone._modules_list.9.bottlenecks.2.cv1.bn.num_batches_tracked",
            "_backbone._modules_list.9.m.2.cv1.bn.running_mean": "_backbone._modules_list.9.bottlenecks.2.cv1.bn.running_mean",
            "_backbone._modules_list.9.m.2.cv1.bn.running_var": "_backbone._modules_list.9.bottlenecks.2.cv1.bn.running_var",
            "_backbone._modules_list.9.m.2.cv1.bn.weight": "_backbone._modules_list.9.bottlenecks.2.cv1.bn.weight",
            "_backbone._modules_list.9.m.2.cv1.conv.weight": "_backbone._modules_list.9.bottlenecks.2.cv1.conv.weight",
            "_backbone._modules_list.9.m.2.cv2.bn.bias": "_backbone._modules_list.9.bottlenecks.2.cv2.bn.bias",
            "_backbone._modules_list.9.m.2.cv2.bn.num_batches_tracked": "_backbone._modules_list.9.bottlenecks.2.cv2.bn.num_batches_tracked",
            "_backbone._modules_list.9.m.2.cv2.bn.running_mean": "_backbone._modules_list.9.bottlenecks.2.cv2.bn.running_mean",
            "_backbone._modules_list.9.m.2.cv2.bn.running_var": "_backbone._modules_list.9.bottlenecks.2.cv2.bn.running_var",
            "_backbone._modules_list.9.m.2.cv2.bn.weight": "_backbone._modules_list.9.bottlenecks.2.cv2.bn.weight",
            "_backbone._modules_list.9.m.2.cv2.conv.weight": "_backbone._modules_list.9.bottlenecks.2.cv2.conv.weight",
            "_head._modules_list.0.bn.bias": "_head._modules_list.0.bn.bias",
            "_head._modules_list.0.bn.num_batches_tracked": "_head._modules_list.0.bn.num_batches_tracked",
            "_head._modules_list.0.bn.running_mean": "_head._modules_list.0.bn.running_mean",
            "_head._modules_list.0.bn.running_var": "_head._modules_list.0.bn.running_var",
            "_head._modules_list.0.bn.weight": "_head._modules_list.0.bn.weight",
            "_head._modules_list.0.conv.weight": "_head._modules_list.0.conv.weight",
            "_head._modules_list.10.cv1.bn.bias": "_head._modules_list.10.conv1.bn.bias",
            "_head._modules_list.10.cv1.bn.num_batches_tracked": "_head._modules_list.10.conv1.bn.num_batches_tracked",
            "_head._modules_list.10.cv1.bn.running_mean": "_head._modules_list.10.conv1.bn.running_mean",
            "_head._modules_list.10.cv1.bn.running_var": "_head._modules_list.10.conv1.bn.running_var",
            "_head._modules_list.10.cv1.bn.weight": "_head._modules_list.10.conv1.bn.weight",
            "_head._modules_list.10.cv1.conv.weight": "_head._modules_list.10.conv1.conv.weight",
            "_head._modules_list.10.cv2.bn.bias": "_head._modules_list.10.conv2.bn.bias",
            "_head._modules_list.10.cv2.bn.num_batches_tracked": "_head._modules_list.10.conv2.bn.num_batches_tracked",
            "_head._modules_list.10.cv2.bn.running_mean": "_head._modules_list.10.conv2.bn.running_mean",
            "_head._modules_list.10.cv2.bn.running_var": "_head._modules_list.10.conv2.bn.running_var",
            "_head._modules_list.10.cv2.bn.weight": "_head._modules_list.10.conv2.bn.weight",
            "_head._modules_list.10.cv2.conv.weight": "_head._modules_list.10.conv2.conv.weight",
            "_head._modules_list.10.cv3.bn.bias": "_head._modules_list.10.conv3.bn.bias",
            "_head._modules_list.10.cv3.bn.num_batches_tracked": "_head._modules_list.10.conv3.bn.num_batches_tracked",
            "_head._modules_list.10.cv3.bn.running_mean": "_head._modules_list.10.conv3.bn.running_mean",
            "_head._modules_list.10.cv3.bn.running_var": "_head._modules_list.10.conv3.bn.running_var",
            "_head._modules_list.10.cv3.bn.weight": "_head._modules_list.10.conv3.bn.weight",
            "_head._modules_list.10.cv3.conv.weight": "_head._modules_list.10.conv3.conv.weight",
            "_head._modules_list.10.m.0.cv1.bn.bias": "_head._modules_list.10.bottlenecks.0.cv1.bn.bias",
            "_head._modules_list.10.m.0.cv1.bn.num_batches_tracked": "_head._modules_list.10.bottlenecks.0.cv1.bn.num_batches_tracked",
            "_head._modules_list.10.m.0.cv1.bn.running_mean": "_head._modules_list.10.bottlenecks.0.cv1.bn.running_mean",
            "_head._modules_list.10.m.0.cv1.bn.running_var": "_head._modules_list.10.bottlenecks.0.cv1.bn.running_var",
            "_head._modules_list.10.m.0.cv1.bn.weight": "_head._modules_list.10.bottlenecks.0.cv1.bn.weight",
            "_head._modules_list.10.m.0.cv1.conv.weight": "_head._modules_list.10.bottlenecks.0.cv1.conv.weight",
            "_head._modules_list.10.m.0.cv2.bn.bias": "_head._modules_list.10.bottlenecks.0.cv2.bn.bias",
            "_head._modules_list.10.m.0.cv2.bn.num_batches_tracked": "_head._modules_list.10.bottlenecks.0.cv2.bn.num_batches_tracked",
            "_head._modules_list.10.m.0.cv2.bn.running_mean": "_head._modules_list.10.bottlenecks.0.cv2.bn.running_mean",
            "_head._modules_list.10.m.0.cv2.bn.running_var": "_head._modules_list.10.bottlenecks.0.cv2.bn.running_var",
            "_head._modules_list.10.m.0.cv2.bn.weight": "_head._modules_list.10.bottlenecks.0.cv2.bn.weight",
            "_head._modules_list.10.m.0.cv2.conv.bn.bias": "_head._modules_list.10.bottlenecks.0.cv2.conv.bn.bias",
            "_head._modules_list.10.m.0.cv2.conv.bn.num_batches_tracked": "_head._modules_list.10.bottlenecks.0.cv2.conv.bn.num_batches_tracked",
            "_head._modules_list.10.m.0.cv2.conv.bn.running_mean": "_head._modules_list.10.bottlenecks.0.cv2.conv.bn.running_mean",
            "_head._modules_list.10.m.0.cv2.conv.bn.running_var": "_head._modules_list.10.bottlenecks.0.cv2.conv.bn.running_var",
            "_head._modules_list.10.m.0.cv2.conv.bn.weight": "_head._modules_list.10.bottlenecks.0.cv2.conv.bn.weight",
            "_head._modules_list.10.m.0.cv2.conv.conv.weight": "_head._modules_list.10.bottlenecks.0.cv2.conv.conv.weight",
            "_head._modules_list.10.m.0.cv2.conv.weight": "_head._modules_list.10.bottlenecks.0.cv2.conv.weight",
            "_head._modules_list.10.m.0.cv2.dconv.bn.bias": "_head._modules_list.10.bottlenecks.0.cv2.dconv.bn.bias",
            "_head._modules_list.10.m.0.cv2.dconv.bn.num_batches_tracked": "_head._modules_list.10.bottlenecks.0.cv2.dconv.bn.num_batches_tracked",
            "_head._modules_list.10.m.0.cv2.dconv.bn.running_mean": "_head._modules_list.10.bottlenecks.0.cv2.dconv.bn.running_mean",
            "_head._modules_list.10.m.0.cv2.dconv.bn.running_var": "_head._modules_list.10.bottlenecks.0.cv2.dconv.bn.running_var",
            "_head._modules_list.10.m.0.cv2.dconv.bn.weight": "_head._modules_list.10.bottlenecks.0.cv2.dconv.bn.weight",
            "_head._modules_list.10.m.0.cv2.dconv.conv.weight": "_head._modules_list.10.bottlenecks.0.cv2.dconv.conv.weight",
            "_head._modules_list.10.m.1.cv1.bn.bias": "_head._modules_list.10.bottlenecks.1.cv1.bn.bias",
            "_head._modules_list.10.m.1.cv1.bn.num_batches_tracked": "_head._modules_list.10.bottlenecks.1.cv1.bn.num_batches_tracked",
            "_head._modules_list.10.m.1.cv1.bn.running_mean": "_head._modules_list.10.bottlenecks.1.cv1.bn.running_mean",
            "_head._modules_list.10.m.1.cv1.bn.running_var": "_head._modules_list.10.bottlenecks.1.cv1.bn.running_var",
            "_head._modules_list.10.m.1.cv1.bn.weight": "_head._modules_list.10.bottlenecks.1.cv1.bn.weight",
            "_head._modules_list.10.m.1.cv1.conv.weight": "_head._modules_list.10.bottlenecks.1.cv1.conv.weight",
            "_head._modules_list.10.m.1.cv2.bn.bias": "_head._modules_list.10.bottlenecks.1.cv2.bn.bias",
            "_head._modules_list.10.m.1.cv2.bn.num_batches_tracked": "_head._modules_list.10.bottlenecks.1.cv2.bn.num_batches_tracked",
            "_head._modules_list.10.m.1.cv2.bn.running_mean": "_head._modules_list.10.bottlenecks.1.cv2.bn.running_mean",
            "_head._modules_list.10.m.1.cv2.bn.running_var": "_head._modules_list.10.bottlenecks.1.cv2.bn.running_var",
            "_head._modules_list.10.m.1.cv2.bn.weight": "_head._modules_list.10.bottlenecks.1.cv2.bn.weight",
            "_head._modules_list.10.m.1.cv2.conv.weight": "_head._modules_list.10.bottlenecks.1.cv2.conv.weight",
            "_head._modules_list.10.m.2.cv1.bn.bias": "_head._modules_list.10.bottlenecks.2.cv1.bn.bias",
            "_head._modules_list.10.m.2.cv1.bn.num_batches_tracked": "_head._modules_list.10.bottlenecks.2.cv1.bn.num_batches_tracked",
            "_head._modules_list.10.m.2.cv1.bn.running_mean": "_head._modules_list.10.bottlenecks.2.cv1.bn.running_mean",
            "_head._modules_list.10.m.2.cv1.bn.running_var": "_head._modules_list.10.bottlenecks.2.cv1.bn.running_var",
            "_head._modules_list.10.m.2.cv1.bn.weight": "_head._modules_list.10.bottlenecks.2.cv1.bn.weight",
            "_head._modules_list.10.m.2.cv1.conv.weight": "_head._modules_list.10.bottlenecks.2.cv1.conv.weight",
            "_head._modules_list.10.m.2.cv2.bn.bias": "_head._modules_list.10.bottlenecks.2.cv2.bn.bias",
            "_head._modules_list.10.m.2.cv2.bn.num_batches_tracked": "_head._modules_list.10.bottlenecks.2.cv2.bn.num_batches_tracked",
            "_head._modules_list.10.m.2.cv2.bn.running_mean": "_head._modules_list.10.bottlenecks.2.cv2.bn.running_mean",
            "_head._modules_list.10.m.2.cv2.bn.running_var": "_head._modules_list.10.bottlenecks.2.cv2.bn.running_var",
            "_head._modules_list.10.m.2.cv2.bn.weight": "_head._modules_list.10.bottlenecks.2.cv2.bn.weight",
            "_head._modules_list.10.m.2.cv2.conv.weight": "_head._modules_list.10.bottlenecks.2.cv2.conv.weight",
            "_head._modules_list.11.bn.bias": "_head._modules_list.11.bn.bias",
            "_head._modules_list.11.bn.num_batches_tracked": "_head._modules_list.11.bn.num_batches_tracked",
            "_head._modules_list.11.bn.running_mean": "_head._modules_list.11.bn.running_mean",
            "_head._modules_list.11.bn.running_var": "_head._modules_list.11.bn.running_var",
            "_head._modules_list.11.bn.weight": "_head._modules_list.11.bn.weight",
            "_head._modules_list.11.conv.bn.bias": "_head._modules_list.11.conv.bn.bias",
            "_head._modules_list.11.conv.bn.num_batches_tracked": "_head._modules_list.11.conv.bn.num_batches_tracked",
            "_head._modules_list.11.conv.bn.running_mean": "_head._modules_list.11.conv.bn.running_mean",
            "_head._modules_list.11.conv.bn.running_var": "_head._modules_list.11.conv.bn.running_var",
            "_head._modules_list.11.conv.bn.weight": "_head._modules_list.11.conv.bn.weight",
            "_head._modules_list.11.conv.conv.weight": "_head._modules_list.11.conv.conv.weight",
            "_head._modules_list.11.conv.weight": "_head._modules_list.11.conv.weight",
            "_head._modules_list.11.dconv.bn.bias": "_head._modules_list.11.dconv.bn.bias",
            "_head._modules_list.11.dconv.bn.num_batches_tracked": "_head._modules_list.11.dconv.bn.num_batches_tracked",
            "_head._modules_list.11.dconv.bn.running_mean": "_head._modules_list.11.dconv.bn.running_mean",
            "_head._modules_list.11.dconv.bn.running_var": "_head._modules_list.11.dconv.bn.running_var",
            "_head._modules_list.11.dconv.bn.weight": "_head._modules_list.11.dconv.bn.weight",
            "_head._modules_list.11.dconv.conv.weight": "_head._modules_list.11.dconv.conv.weight",
            "_head._modules_list.13.cv1.bn.bias": "_head._modules_list.13.conv1.bn.bias",
            "_head._modules_list.13.cv1.bn.num_batches_tracked": "_head._modules_list.13.conv1.bn.num_batches_tracked",
            "_head._modules_list.13.cv1.bn.running_mean": "_head._modules_list.13.conv1.bn.running_mean",
            "_head._modules_list.13.cv1.bn.running_var": "_head._modules_list.13.conv1.bn.running_var",
            "_head._modules_list.13.cv1.bn.weight": "_head._modules_list.13.conv1.bn.weight",
            "_head._modules_list.13.cv1.conv.weight": "_head._modules_list.13.conv1.conv.weight",
            "_head._modules_list.13.cv2.bn.bias": "_head._modules_list.13.conv2.bn.bias",
            "_head._modules_list.13.cv2.bn.num_batches_tracked": "_head._modules_list.13.conv2.bn.num_batches_tracked",
            "_head._modules_list.13.cv2.bn.running_mean": "_head._modules_list.13.conv2.bn.running_mean",
            "_head._modules_list.13.cv2.bn.running_var": "_head._modules_list.13.conv2.bn.running_var",
            "_head._modules_list.13.cv2.bn.weight": "_head._modules_list.13.conv2.bn.weight",
            "_head._modules_list.13.cv2.conv.weight": "_head._modules_list.13.conv2.conv.weight",
            "_head._modules_list.13.cv3.bn.bias": "_head._modules_list.13.conv3.bn.bias",
            "_head._modules_list.13.cv3.bn.num_batches_tracked": "_head._modules_list.13.conv3.bn.num_batches_tracked",
            "_head._modules_list.13.cv3.bn.running_mean": "_head._modules_list.13.conv3.bn.running_mean",
            "_head._modules_list.13.cv3.bn.running_var": "_head._modules_list.13.conv3.bn.running_var",
            "_head._modules_list.13.cv3.bn.weight": "_head._modules_list.13.conv3.bn.weight",
            "_head._modules_list.13.cv3.conv.weight": "_head._modules_list.13.conv3.conv.weight",
            "_head._modules_list.13.m.0.cv1.bn.bias": "_head._modules_list.13.bottlenecks.0.cv1.bn.bias",
            "_head._modules_list.13.m.0.cv1.bn.num_batches_tracked": "_head._modules_list.13.bottlenecks.0.cv1.bn.num_batches_tracked",
            "_head._modules_list.13.m.0.cv1.bn.running_mean": "_head._modules_list.13.bottlenecks.0.cv1.bn.running_mean",
            "_head._modules_list.13.m.0.cv1.bn.running_var": "_head._modules_list.13.bottlenecks.0.cv1.bn.running_var",
            "_head._modules_list.13.m.0.cv1.bn.weight": "_head._modules_list.13.bottlenecks.0.cv1.bn.weight",
            "_head._modules_list.13.m.0.cv1.conv.weight": "_head._modules_list.13.bottlenecks.0.cv1.conv.weight",
            "_head._modules_list.13.m.0.cv2.bn.bias": "_head._modules_list.13.bottlenecks.0.cv2.bn.bias",
            "_head._modules_list.13.m.0.cv2.bn.num_batches_tracked": "_head._modules_list.13.bottlenecks.0.cv2.bn.num_batches_tracked",
            "_head._modules_list.13.m.0.cv2.bn.running_mean": "_head._modules_list.13.bottlenecks.0.cv2.bn.running_mean",
            "_head._modules_list.13.m.0.cv2.bn.running_var": "_head._modules_list.13.bottlenecks.0.cv2.bn.running_var",
            "_head._modules_list.13.m.0.cv2.bn.weight": "_head._modules_list.13.bottlenecks.0.cv2.bn.weight",
            "_head._modules_list.13.m.0.cv2.conv.bn.bias": "_head._modules_list.13.bottlenecks.0.cv2.conv.bn.bias",
            "_head._modules_list.13.m.0.cv2.conv.bn.num_batches_tracked": "_head._modules_list.13.bottlenecks.0.cv2.conv.bn.num_batches_tracked",
            "_head._modules_list.13.m.0.cv2.conv.bn.running_mean": "_head._modules_list.13.bottlenecks.0.cv2.conv.bn.running_mean",
            "_head._modules_list.13.m.0.cv2.conv.bn.running_var": "_head._modules_list.13.bottlenecks.0.cv2.conv.bn.running_var",
            "_head._modules_list.13.m.0.cv2.conv.bn.weight": "_head._modules_list.13.bottlenecks.0.cv2.conv.bn.weight",
            "_head._modules_list.13.m.0.cv2.conv.conv.weight": "_head._modules_list.13.bottlenecks.0.cv2.conv.conv.weight",
            "_head._modules_list.13.m.0.cv2.conv.weight": "_head._modules_list.13.bottlenecks.0.cv2.conv.weight",
            "_head._modules_list.13.m.0.cv2.dconv.bn.bias": "_head._modules_list.13.bottlenecks.0.cv2.dconv.bn.bias",
            "_head._modules_list.13.m.0.cv2.dconv.bn.num_batches_tracked": "_head._modules_list.13.bottlenecks.0.cv2.dconv.bn.num_batches_tracked",
            "_head._modules_list.13.m.0.cv2.dconv.bn.running_mean": "_head._modules_list.13.bottlenecks.0.cv2.dconv.bn.running_mean",
            "_head._modules_list.13.m.0.cv2.dconv.bn.running_var": "_head._modules_list.13.bottlenecks.0.cv2.dconv.bn.running_var",
            "_head._modules_list.13.m.0.cv2.dconv.bn.weight": "_head._modules_list.13.bottlenecks.0.cv2.dconv.bn.weight",
            "_head._modules_list.13.m.0.cv2.dconv.conv.weight": "_head._modules_list.13.bottlenecks.0.cv2.dconv.conv.weight",
            "_head._modules_list.13.m.1.cv1.bn.bias": "_head._modules_list.13.bottlenecks.1.cv1.bn.bias",
            "_head._modules_list.13.m.1.cv1.bn.num_batches_tracked": "_head._modules_list.13.bottlenecks.1.cv1.bn.num_batches_tracked",
            "_head._modules_list.13.m.1.cv1.bn.running_mean": "_head._modules_list.13.bottlenecks.1.cv1.bn.running_mean",
            "_head._modules_list.13.m.1.cv1.bn.running_var": "_head._modules_list.13.bottlenecks.1.cv1.bn.running_var",
            "_head._modules_list.13.m.1.cv1.bn.weight": "_head._modules_list.13.bottlenecks.1.cv1.bn.weight",
            "_head._modules_list.13.m.1.cv1.conv.weight": "_head._modules_list.13.bottlenecks.1.cv1.conv.weight",
            "_head._modules_list.13.m.1.cv2.bn.bias": "_head._modules_list.13.bottlenecks.1.cv2.bn.bias",
            "_head._modules_list.13.m.1.cv2.bn.num_batches_tracked": "_head._modules_list.13.bottlenecks.1.cv2.bn.num_batches_tracked",
            "_head._modules_list.13.m.1.cv2.bn.running_mean": "_head._modules_list.13.bottlenecks.1.cv2.bn.running_mean",
            "_head._modules_list.13.m.1.cv2.bn.running_var": "_head._modules_list.13.bottlenecks.1.cv2.bn.running_var",
            "_head._modules_list.13.m.1.cv2.bn.weight": "_head._modules_list.13.bottlenecks.1.cv2.bn.weight",
            "_head._modules_list.13.m.1.cv2.conv.weight": "_head._modules_list.13.bottlenecks.1.cv2.conv.weight",
            "_head._modules_list.13.m.2.cv1.bn.bias": "_head._modules_list.13.bottlenecks.2.cv1.bn.bias",
            "_head._modules_list.13.m.2.cv1.bn.num_batches_tracked": "_head._modules_list.13.bottlenecks.2.cv1.bn.num_batches_tracked",
            "_head._modules_list.13.m.2.cv1.bn.running_mean": "_head._modules_list.13.bottlenecks.2.cv1.bn.running_mean",
            "_head._modules_list.13.m.2.cv1.bn.running_var": "_head._modules_list.13.bottlenecks.2.cv1.bn.running_var",
            "_head._modules_list.13.m.2.cv1.bn.weight": "_head._modules_list.13.bottlenecks.2.cv1.bn.weight",
            "_head._modules_list.13.m.2.cv1.conv.weight": "_head._modules_list.13.bottlenecks.2.cv1.conv.weight",
            "_head._modules_list.13.m.2.cv2.bn.bias": "_head._modules_list.13.bottlenecks.2.cv2.bn.bias",
            "_head._modules_list.13.m.2.cv2.bn.num_batches_tracked": "_head._modules_list.13.bottlenecks.2.cv2.bn.num_batches_tracked",
            "_head._modules_list.13.m.2.cv2.bn.running_mean": "_head._modules_list.13.bottlenecks.2.cv2.bn.running_mean",
            "_head._modules_list.13.m.2.cv2.bn.running_var": "_head._modules_list.13.bottlenecks.2.cv2.bn.running_var",
            "_head._modules_list.13.m.2.cv2.bn.weight": "_head._modules_list.13.bottlenecks.2.cv2.bn.weight",
            "_head._modules_list.13.m.2.cv2.conv.weight": "_head._modules_list.13.bottlenecks.2.cv2.conv.weight",
            "_head._modules_list.14.cls_convs.0.0.bn.bias": "_head._modules_list.14.cls_convs.0.0.bn.bias",
            "_head._modules_list.14.cls_convs.0.0.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.0.0.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.0.0.bn.running_mean": "_head._modules_list.14.cls_convs.0.0.bn.running_mean",
            "_head._modules_list.14.cls_convs.0.0.bn.running_var": "_head._modules_list.14.cls_convs.0.0.bn.running_var",
            "_head._modules_list.14.cls_convs.0.0.bn.weight": "_head._modules_list.14.cls_convs.0.0.bn.weight",
            "_head._modules_list.14.cls_convs.0.0.conv.bn.bias": "_head._modules_list.14.cls_convs.0.0.conv.bn.bias",
            "_head._modules_list.14.cls_convs.0.0.conv.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.0.0.conv.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.0.0.conv.bn.running_mean": "_head._modules_list.14.cls_convs.0.0.conv.bn.running_mean",
            "_head._modules_list.14.cls_convs.0.0.conv.bn.running_var": "_head._modules_list.14.cls_convs.0.0.conv.bn.running_var",
            "_head._modules_list.14.cls_convs.0.0.conv.bn.weight": "_head._modules_list.14.cls_convs.0.0.conv.bn.weight",
            "_head._modules_list.14.cls_convs.0.0.conv.conv.weight": "_head._modules_list.14.cls_convs.0.0.conv.conv.weight",
            "_head._modules_list.14.cls_convs.0.0.conv.weight": "_head._modules_list.14.cls_convs.0.0.conv.weight",
            "_head._modules_list.14.cls_convs.0.0.dconv.bn.bias": "_head._modules_list.14.cls_convs.0.0.dconv.bn.bias",
            "_head._modules_list.14.cls_convs.0.0.dconv.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.0.0.dconv.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.0.0.dconv.bn.running_mean": "_head._modules_list.14.cls_convs.0.0.dconv.bn.running_mean",
            "_head._modules_list.14.cls_convs.0.0.dconv.bn.running_var": "_head._modules_list.14.cls_convs.0.0.dconv.bn.running_var",
            "_head._modules_list.14.cls_convs.0.0.dconv.bn.weight": "_head._modules_list.14.cls_convs.0.0.dconv.bn.weight",
            "_head._modules_list.14.cls_convs.0.0.dconv.conv.weight": "_head._modules_list.14.cls_convs.0.0.dconv.conv.weight",
            "_head._modules_list.14.cls_convs.0.1.bn.bias": "_head._modules_list.14.cls_convs.0.1.bn.bias",
            "_head._modules_list.14.cls_convs.0.1.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.0.1.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.0.1.bn.running_mean": "_head._modules_list.14.cls_convs.0.1.bn.running_mean",
            "_head._modules_list.14.cls_convs.0.1.bn.running_var": "_head._modules_list.14.cls_convs.0.1.bn.running_var",
            "_head._modules_list.14.cls_convs.0.1.bn.weight": "_head._modules_list.14.cls_convs.0.1.bn.weight",
            "_head._modules_list.14.cls_convs.0.1.conv.bn.bias": "_head._modules_list.14.cls_convs.0.1.conv.bn.bias",
            "_head._modules_list.14.cls_convs.0.1.conv.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.0.1.conv.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.0.1.conv.bn.running_mean": "_head._modules_list.14.cls_convs.0.1.conv.bn.running_mean",
            "_head._modules_list.14.cls_convs.0.1.conv.bn.running_var": "_head._modules_list.14.cls_convs.0.1.conv.bn.running_var",
            "_head._modules_list.14.cls_convs.0.1.conv.bn.weight": "_head._modules_list.14.cls_convs.0.1.conv.bn.weight",
            "_head._modules_list.14.cls_convs.0.1.conv.conv.weight": "_head._modules_list.14.cls_convs.0.1.conv.conv.weight",
            "_head._modules_list.14.cls_convs.0.1.conv.weight": "_head._modules_list.14.cls_convs.0.1.conv.weight",
            "_head._modules_list.14.cls_convs.0.1.dconv.bn.bias": "_head._modules_list.14.cls_convs.0.1.dconv.bn.bias",
            "_head._modules_list.14.cls_convs.0.1.dconv.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.0.1.dconv.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.0.1.dconv.bn.running_mean": "_head._modules_list.14.cls_convs.0.1.dconv.bn.running_mean",
            "_head._modules_list.14.cls_convs.0.1.dconv.bn.running_var": "_head._modules_list.14.cls_convs.0.1.dconv.bn.running_var",
            "_head._modules_list.14.cls_convs.0.1.dconv.bn.weight": "_head._modules_list.14.cls_convs.0.1.dconv.bn.weight",
            "_head._modules_list.14.cls_convs.0.1.dconv.conv.weight": "_head._modules_list.14.cls_convs.0.1.dconv.conv.weight",
            "_head._modules_list.14.cls_convs.1.0.bn.bias": "_head._modules_list.14.cls_convs.1.0.bn.bias",
            "_head._modules_list.14.cls_convs.1.0.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.1.0.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.1.0.bn.running_mean": "_head._modules_list.14.cls_convs.1.0.bn.running_mean",
            "_head._modules_list.14.cls_convs.1.0.bn.running_var": "_head._modules_list.14.cls_convs.1.0.bn.running_var",
            "_head._modules_list.14.cls_convs.1.0.bn.weight": "_head._modules_list.14.cls_convs.1.0.bn.weight",
            "_head._modules_list.14.cls_convs.1.0.conv.bn.bias": "_head._modules_list.14.cls_convs.1.0.conv.bn.bias",
            "_head._modules_list.14.cls_convs.1.0.conv.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.1.0.conv.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.1.0.conv.bn.running_mean": "_head._modules_list.14.cls_convs.1.0.conv.bn.running_mean",
            "_head._modules_list.14.cls_convs.1.0.conv.bn.running_var": "_head._modules_list.14.cls_convs.1.0.conv.bn.running_var",
            "_head._modules_list.14.cls_convs.1.0.conv.bn.weight": "_head._modules_list.14.cls_convs.1.0.conv.bn.weight",
            "_head._modules_list.14.cls_convs.1.0.conv.conv.weight": "_head._modules_list.14.cls_convs.1.0.conv.conv.weight",
            "_head._modules_list.14.cls_convs.1.0.conv.weight": "_head._modules_list.14.cls_convs.1.0.conv.weight",
            "_head._modules_list.14.cls_convs.1.0.dconv.bn.bias": "_head._modules_list.14.cls_convs.1.0.dconv.bn.bias",
            "_head._modules_list.14.cls_convs.1.0.dconv.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.1.0.dconv.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.1.0.dconv.bn.running_mean": "_head._modules_list.14.cls_convs.1.0.dconv.bn.running_mean",
            "_head._modules_list.14.cls_convs.1.0.dconv.bn.running_var": "_head._modules_list.14.cls_convs.1.0.dconv.bn.running_var",
            "_head._modules_list.14.cls_convs.1.0.dconv.bn.weight": "_head._modules_list.14.cls_convs.1.0.dconv.bn.weight",
            "_head._modules_list.14.cls_convs.1.0.dconv.conv.weight": "_head._modules_list.14.cls_convs.1.0.dconv.conv.weight",
            "_head._modules_list.14.cls_convs.1.1.bn.bias": "_head._modules_list.14.cls_convs.1.1.bn.bias",
            "_head._modules_list.14.cls_convs.1.1.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.1.1.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.1.1.bn.running_mean": "_head._modules_list.14.cls_convs.1.1.bn.running_mean",
            "_head._modules_list.14.cls_convs.1.1.bn.running_var": "_head._modules_list.14.cls_convs.1.1.bn.running_var",
            "_head._modules_list.14.cls_convs.1.1.bn.weight": "_head._modules_list.14.cls_convs.1.1.bn.weight",
            "_head._modules_list.14.cls_convs.1.1.conv.bn.bias": "_head._modules_list.14.cls_convs.1.1.conv.bn.bias",
            "_head._modules_list.14.cls_convs.1.1.conv.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.1.1.conv.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.1.1.conv.bn.running_mean": "_head._modules_list.14.cls_convs.1.1.conv.bn.running_mean",
            "_head._modules_list.14.cls_convs.1.1.conv.bn.running_var": "_head._modules_list.14.cls_convs.1.1.conv.bn.running_var",
            "_head._modules_list.14.cls_convs.1.1.conv.bn.weight": "_head._modules_list.14.cls_convs.1.1.conv.bn.weight",
            "_head._modules_list.14.cls_convs.1.1.conv.conv.weight": "_head._modules_list.14.cls_convs.1.1.conv.conv.weight",
            "_head._modules_list.14.cls_convs.1.1.conv.weight": "_head._modules_list.14.cls_convs.1.1.conv.weight",
            "_head._modules_list.14.cls_convs.1.1.dconv.bn.bias": "_head._modules_list.14.cls_convs.1.1.dconv.bn.bias",
            "_head._modules_list.14.cls_convs.1.1.dconv.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.1.1.dconv.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.1.1.dconv.bn.running_mean": "_head._modules_list.14.cls_convs.1.1.dconv.bn.running_mean",
            "_head._modules_list.14.cls_convs.1.1.dconv.bn.running_var": "_head._modules_list.14.cls_convs.1.1.dconv.bn.running_var",
            "_head._modules_list.14.cls_convs.1.1.dconv.bn.weight": "_head._modules_list.14.cls_convs.1.1.dconv.bn.weight",
            "_head._modules_list.14.cls_convs.1.1.dconv.conv.weight": "_head._modules_list.14.cls_convs.1.1.dconv.conv.weight",
            "_head._modules_list.14.cls_convs.2.0.bn.bias": "_head._modules_list.14.cls_convs.2.0.bn.bias",
            "_head._modules_list.14.cls_convs.2.0.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.2.0.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.2.0.bn.running_mean": "_head._modules_list.14.cls_convs.2.0.bn.running_mean",
            "_head._modules_list.14.cls_convs.2.0.bn.running_var": "_head._modules_list.14.cls_convs.2.0.bn.running_var",
            "_head._modules_list.14.cls_convs.2.0.bn.weight": "_head._modules_list.14.cls_convs.2.0.bn.weight",
            "_head._modules_list.14.cls_convs.2.0.conv.bn.bias": "_head._modules_list.14.cls_convs.2.0.conv.bn.bias",
            "_head._modules_list.14.cls_convs.2.0.conv.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.2.0.conv.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.2.0.conv.bn.running_mean": "_head._modules_list.14.cls_convs.2.0.conv.bn.running_mean",
            "_head._modules_list.14.cls_convs.2.0.conv.bn.running_var": "_head._modules_list.14.cls_convs.2.0.conv.bn.running_var",
            "_head._modules_list.14.cls_convs.2.0.conv.bn.weight": "_head._modules_list.14.cls_convs.2.0.conv.bn.weight",
            "_head._modules_list.14.cls_convs.2.0.conv.conv.weight": "_head._modules_list.14.cls_convs.2.0.conv.conv.weight",
            "_head._modules_list.14.cls_convs.2.0.conv.weight": "_head._modules_list.14.cls_convs.2.0.conv.weight",
            "_head._modules_list.14.cls_convs.2.0.dconv.bn.bias": "_head._modules_list.14.cls_convs.2.0.dconv.bn.bias",
            "_head._modules_list.14.cls_convs.2.0.dconv.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.2.0.dconv.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.2.0.dconv.bn.running_mean": "_head._modules_list.14.cls_convs.2.0.dconv.bn.running_mean",
            "_head._modules_list.14.cls_convs.2.0.dconv.bn.running_var": "_head._modules_list.14.cls_convs.2.0.dconv.bn.running_var",
            "_head._modules_list.14.cls_convs.2.0.dconv.bn.weight": "_head._modules_list.14.cls_convs.2.0.dconv.bn.weight",
            "_head._modules_list.14.cls_convs.2.0.dconv.conv.weight": "_head._modules_list.14.cls_convs.2.0.dconv.conv.weight",
            "_head._modules_list.14.cls_convs.2.1.bn.bias": "_head._modules_list.14.cls_convs.2.1.bn.bias",
            "_head._modules_list.14.cls_convs.2.1.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.2.1.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.2.1.bn.running_mean": "_head._modules_list.14.cls_convs.2.1.bn.running_mean",
            "_head._modules_list.14.cls_convs.2.1.bn.running_var": "_head._modules_list.14.cls_convs.2.1.bn.running_var",
            "_head._modules_list.14.cls_convs.2.1.bn.weight": "_head._modules_list.14.cls_convs.2.1.bn.weight",
            "_head._modules_list.14.cls_convs.2.1.conv.bn.bias": "_head._modules_list.14.cls_convs.2.1.conv.bn.bias",
            "_head._modules_list.14.cls_convs.2.1.conv.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.2.1.conv.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.2.1.conv.bn.running_mean": "_head._modules_list.14.cls_convs.2.1.conv.bn.running_mean",
            "_head._modules_list.14.cls_convs.2.1.conv.bn.running_var": "_head._modules_list.14.cls_convs.2.1.conv.bn.running_var",
            "_head._modules_list.14.cls_convs.2.1.conv.bn.weight": "_head._modules_list.14.cls_convs.2.1.conv.bn.weight",
            "_head._modules_list.14.cls_convs.2.1.conv.conv.weight": "_head._modules_list.14.cls_convs.2.1.conv.conv.weight",
            "_head._modules_list.14.cls_convs.2.1.conv.weight": "_head._modules_list.14.cls_convs.2.1.conv.weight",
            "_head._modules_list.14.cls_convs.2.1.dconv.bn.bias": "_head._modules_list.14.cls_convs.2.1.dconv.bn.bias",
            "_head._modules_list.14.cls_convs.2.1.dconv.bn.num_batches_tracked": "_head._modules_list.14.cls_convs.2.1.dconv.bn.num_batches_tracked",
            "_head._modules_list.14.cls_convs.2.1.dconv.bn.running_mean": "_head._modules_list.14.cls_convs.2.1.dconv.bn.running_mean",
            "_head._modules_list.14.cls_convs.2.1.dconv.bn.running_var": "_head._modules_list.14.cls_convs.2.1.dconv.bn.running_var",
            "_head._modules_list.14.cls_convs.2.1.dconv.bn.weight": "_head._modules_list.14.cls_convs.2.1.dconv.bn.weight",
            "_head._modules_list.14.cls_convs.2.1.dconv.conv.weight": "_head._modules_list.14.cls_convs.2.1.dconv.conv.weight",
            "_head._modules_list.14.cls_preds.0.bias": "_head._modules_list.14.cls_preds.0.bias",
            "_head._modules_list.14.cls_preds.0.weight": "_head._modules_list.14.cls_preds.0.weight",
            "_head._modules_list.14.cls_preds.1.bias": "_head._modules_list.14.cls_preds.1.bias",
            "_head._modules_list.14.cls_preds.1.weight": "_head._modules_list.14.cls_preds.1.weight",
            "_head._modules_list.14.cls_preds.2.bias": "_head._modules_list.14.cls_preds.2.bias",
            "_head._modules_list.14.cls_preds.2.weight": "_head._modules_list.14.cls_preds.2.weight",
            "_head._modules_list.14.obj_preds.0.bias": "_head._modules_list.14.obj_preds.0.bias",
            "_head._modules_list.14.obj_preds.0.weight": "_head._modules_list.14.obj_preds.0.weight",
            "_head._modules_list.14.obj_preds.1.bias": "_head._modules_list.14.obj_preds.1.bias",
            "_head._modules_list.14.obj_preds.1.weight": "_head._modules_list.14.obj_preds.1.weight",
            "_head._modules_list.14.obj_preds.2.bias": "_head._modules_list.14.obj_preds.2.bias",
            "_head._modules_list.14.obj_preds.2.weight": "_head._modules_list.14.obj_preds.2.weight",
            "_head._modules_list.14.reg_convs.0.0.bn.bias": "_head._modules_list.14.reg_convs.0.0.bn.bias",
            "_head._modules_list.14.reg_convs.0.0.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.0.0.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.0.0.bn.running_mean": "_head._modules_list.14.reg_convs.0.0.bn.running_mean",
            "_head._modules_list.14.reg_convs.0.0.bn.running_var": "_head._modules_list.14.reg_convs.0.0.bn.running_var",
            "_head._modules_list.14.reg_convs.0.0.bn.weight": "_head._modules_list.14.reg_convs.0.0.bn.weight",
            "_head._modules_list.14.reg_convs.0.0.conv.bn.bias": "_head._modules_list.14.reg_convs.0.0.conv.bn.bias",
            "_head._modules_list.14.reg_convs.0.0.conv.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.0.0.conv.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.0.0.conv.bn.running_mean": "_head._modules_list.14.reg_convs.0.0.conv.bn.running_mean",
            "_head._modules_list.14.reg_convs.0.0.conv.bn.running_var": "_head._modules_list.14.reg_convs.0.0.conv.bn.running_var",
            "_head._modules_list.14.reg_convs.0.0.conv.bn.weight": "_head._modules_list.14.reg_convs.0.0.conv.bn.weight",
            "_head._modules_list.14.reg_convs.0.0.conv.conv.weight": "_head._modules_list.14.reg_convs.0.0.conv.conv.weight",
            "_head._modules_list.14.reg_convs.0.0.conv.weight": "_head._modules_list.14.reg_convs.0.0.conv.weight",
            "_head._modules_list.14.reg_convs.0.0.dconv.bn.bias": "_head._modules_list.14.reg_convs.0.0.dconv.bn.bias",
            "_head._modules_list.14.reg_convs.0.0.dconv.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.0.0.dconv.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.0.0.dconv.bn.running_mean": "_head._modules_list.14.reg_convs.0.0.dconv.bn.running_mean",
            "_head._modules_list.14.reg_convs.0.0.dconv.bn.running_var": "_head._modules_list.14.reg_convs.0.0.dconv.bn.running_var",
            "_head._modules_list.14.reg_convs.0.0.dconv.bn.weight": "_head._modules_list.14.reg_convs.0.0.dconv.bn.weight",
            "_head._modules_list.14.reg_convs.0.0.dconv.conv.weight": "_head._modules_list.14.reg_convs.0.0.dconv.conv.weight",
            "_head._modules_list.14.reg_convs.0.1.bn.bias": "_head._modules_list.14.reg_convs.0.1.bn.bias",
            "_head._modules_list.14.reg_convs.0.1.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.0.1.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.0.1.bn.running_mean": "_head._modules_list.14.reg_convs.0.1.bn.running_mean",
            "_head._modules_list.14.reg_convs.0.1.bn.running_var": "_head._modules_list.14.reg_convs.0.1.bn.running_var",
            "_head._modules_list.14.reg_convs.0.1.bn.weight": "_head._modules_list.14.reg_convs.0.1.bn.weight",
            "_head._modules_list.14.reg_convs.0.1.conv.bn.bias": "_head._modules_list.14.reg_convs.0.1.conv.bn.bias",
            "_head._modules_list.14.reg_convs.0.1.conv.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.0.1.conv.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.0.1.conv.bn.running_mean": "_head._modules_list.14.reg_convs.0.1.conv.bn.running_mean",
            "_head._modules_list.14.reg_convs.0.1.conv.bn.running_var": "_head._modules_list.14.reg_convs.0.1.conv.bn.running_var",
            "_head._modules_list.14.reg_convs.0.1.conv.bn.weight": "_head._modules_list.14.reg_convs.0.1.conv.bn.weight",
            "_head._modules_list.14.reg_convs.0.1.conv.conv.weight": "_head._modules_list.14.reg_convs.0.1.conv.conv.weight",
            "_head._modules_list.14.reg_convs.0.1.conv.weight": "_head._modules_list.14.reg_convs.0.1.conv.weight",
            "_head._modules_list.14.reg_convs.0.1.dconv.bn.bias": "_head._modules_list.14.reg_convs.0.1.dconv.bn.bias",
            "_head._modules_list.14.reg_convs.0.1.dconv.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.0.1.dconv.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.0.1.dconv.bn.running_mean": "_head._modules_list.14.reg_convs.0.1.dconv.bn.running_mean",
            "_head._modules_list.14.reg_convs.0.1.dconv.bn.running_var": "_head._modules_list.14.reg_convs.0.1.dconv.bn.running_var",
            "_head._modules_list.14.reg_convs.0.1.dconv.bn.weight": "_head._modules_list.14.reg_convs.0.1.dconv.bn.weight",
            "_head._modules_list.14.reg_convs.0.1.dconv.conv.weight": "_head._modules_list.14.reg_convs.0.1.dconv.conv.weight",
            "_head._modules_list.14.reg_convs.1.0.bn.bias": "_head._modules_list.14.reg_convs.1.0.bn.bias",
            "_head._modules_list.14.reg_convs.1.0.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.1.0.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.1.0.bn.running_mean": "_head._modules_list.14.reg_convs.1.0.bn.running_mean",
            "_head._modules_list.14.reg_convs.1.0.bn.running_var": "_head._modules_list.14.reg_convs.1.0.bn.running_var",
            "_head._modules_list.14.reg_convs.1.0.bn.weight": "_head._modules_list.14.reg_convs.1.0.bn.weight",
            "_head._modules_list.14.reg_convs.1.0.conv.bn.bias": "_head._modules_list.14.reg_convs.1.0.conv.bn.bias",
            "_head._modules_list.14.reg_convs.1.0.conv.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.1.0.conv.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.1.0.conv.bn.running_mean": "_head._modules_list.14.reg_convs.1.0.conv.bn.running_mean",
            "_head._modules_list.14.reg_convs.1.0.conv.bn.running_var": "_head._modules_list.14.reg_convs.1.0.conv.bn.running_var",
            "_head._modules_list.14.reg_convs.1.0.conv.bn.weight": "_head._modules_list.14.reg_convs.1.0.conv.bn.weight",
            "_head._modules_list.14.reg_convs.1.0.conv.conv.weight": "_head._modules_list.14.reg_convs.1.0.conv.conv.weight",
            "_head._modules_list.14.reg_convs.1.0.conv.weight": "_head._modules_list.14.reg_convs.1.0.conv.weight",
            "_head._modules_list.14.reg_convs.1.0.dconv.bn.bias": "_head._modules_list.14.reg_convs.1.0.dconv.bn.bias",
            "_head._modules_list.14.reg_convs.1.0.dconv.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.1.0.dconv.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.1.0.dconv.bn.running_mean": "_head._modules_list.14.reg_convs.1.0.dconv.bn.running_mean",
            "_head._modules_list.14.reg_convs.1.0.dconv.bn.running_var": "_head._modules_list.14.reg_convs.1.0.dconv.bn.running_var",
            "_head._modules_list.14.reg_convs.1.0.dconv.bn.weight": "_head._modules_list.14.reg_convs.1.0.dconv.bn.weight",
            "_head._modules_list.14.reg_convs.1.0.dconv.conv.weight": "_head._modules_list.14.reg_convs.1.0.dconv.conv.weight",
            "_head._modules_list.14.reg_convs.1.1.bn.bias": "_head._modules_list.14.reg_convs.1.1.bn.bias",
            "_head._modules_list.14.reg_convs.1.1.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.1.1.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.1.1.bn.running_mean": "_head._modules_list.14.reg_convs.1.1.bn.running_mean",
            "_head._modules_list.14.reg_convs.1.1.bn.running_var": "_head._modules_list.14.reg_convs.1.1.bn.running_var",
            "_head._modules_list.14.reg_convs.1.1.bn.weight": "_head._modules_list.14.reg_convs.1.1.bn.weight",
            "_head._modules_list.14.reg_convs.1.1.conv.bn.bias": "_head._modules_list.14.reg_convs.1.1.conv.bn.bias",
            "_head._modules_list.14.reg_convs.1.1.conv.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.1.1.conv.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.1.1.conv.bn.running_mean": "_head._modules_list.14.reg_convs.1.1.conv.bn.running_mean",
            "_head._modules_list.14.reg_convs.1.1.conv.bn.running_var": "_head._modules_list.14.reg_convs.1.1.conv.bn.running_var",
            "_head._modules_list.14.reg_convs.1.1.conv.bn.weight": "_head._modules_list.14.reg_convs.1.1.conv.bn.weight",
            "_head._modules_list.14.reg_convs.1.1.conv.conv.weight": "_head._modules_list.14.reg_convs.1.1.conv.conv.weight",
            "_head._modules_list.14.reg_convs.1.1.conv.weight": "_head._modules_list.14.reg_convs.1.1.conv.weight",
            "_head._modules_list.14.reg_convs.1.1.dconv.bn.bias": "_head._modules_list.14.reg_convs.1.1.dconv.bn.bias",
            "_head._modules_list.14.reg_convs.1.1.dconv.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.1.1.dconv.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.1.1.dconv.bn.running_mean": "_head._modules_list.14.reg_convs.1.1.dconv.bn.running_mean",
            "_head._modules_list.14.reg_convs.1.1.dconv.bn.running_var": "_head._modules_list.14.reg_convs.1.1.dconv.bn.running_var",
            "_head._modules_list.14.reg_convs.1.1.dconv.bn.weight": "_head._modules_list.14.reg_convs.1.1.dconv.bn.weight",
            "_head._modules_list.14.reg_convs.1.1.dconv.conv.weight": "_head._modules_list.14.reg_convs.1.1.dconv.conv.weight",
            "_head._modules_list.14.reg_convs.2.0.bn.bias": "_head._modules_list.14.reg_convs.2.0.bn.bias",
            "_head._modules_list.14.reg_convs.2.0.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.2.0.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.2.0.bn.running_mean": "_head._modules_list.14.reg_convs.2.0.bn.running_mean",
            "_head._modules_list.14.reg_convs.2.0.bn.running_var": "_head._modules_list.14.reg_convs.2.0.bn.running_var",
            "_head._modules_list.14.reg_convs.2.0.bn.weight": "_head._modules_list.14.reg_convs.2.0.bn.weight",
            "_head._modules_list.14.reg_convs.2.0.conv.bn.bias": "_head._modules_list.14.reg_convs.2.0.conv.bn.bias",
            "_head._modules_list.14.reg_convs.2.0.conv.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.2.0.conv.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.2.0.conv.bn.running_mean": "_head._modules_list.14.reg_convs.2.0.conv.bn.running_mean",
            "_head._modules_list.14.reg_convs.2.0.conv.bn.running_var": "_head._modules_list.14.reg_convs.2.0.conv.bn.running_var",
            "_head._modules_list.14.reg_convs.2.0.conv.bn.weight": "_head._modules_list.14.reg_convs.2.0.conv.bn.weight",
            "_head._modules_list.14.reg_convs.2.0.conv.conv.weight": "_head._modules_list.14.reg_convs.2.0.conv.conv.weight",
            "_head._modules_list.14.reg_convs.2.0.conv.weight": "_head._modules_list.14.reg_convs.2.0.conv.weight",
            "_head._modules_list.14.reg_convs.2.0.dconv.bn.bias": "_head._modules_list.14.reg_convs.2.0.dconv.bn.bias",
            "_head._modules_list.14.reg_convs.2.0.dconv.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.2.0.dconv.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.2.0.dconv.bn.running_mean": "_head._modules_list.14.reg_convs.2.0.dconv.bn.running_mean",
            "_head._modules_list.14.reg_convs.2.0.dconv.bn.running_var": "_head._modules_list.14.reg_convs.2.0.dconv.bn.running_var",
            "_head._modules_list.14.reg_convs.2.0.dconv.bn.weight": "_head._modules_list.14.reg_convs.2.0.dconv.bn.weight",
            "_head._modules_list.14.reg_convs.2.0.dconv.conv.weight": "_head._modules_list.14.reg_convs.2.0.dconv.conv.weight",
            "_head._modules_list.14.reg_convs.2.1.bn.bias": "_head._modules_list.14.reg_convs.2.1.bn.bias",
            "_head._modules_list.14.reg_convs.2.1.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.2.1.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.2.1.bn.running_mean": "_head._modules_list.14.reg_convs.2.1.bn.running_mean",
            "_head._modules_list.14.reg_convs.2.1.bn.running_var": "_head._modules_list.14.reg_convs.2.1.bn.running_var",
            "_head._modules_list.14.reg_convs.2.1.bn.weight": "_head._modules_list.14.reg_convs.2.1.bn.weight",
            "_head._modules_list.14.reg_convs.2.1.conv.bn.bias": "_head._modules_list.14.reg_convs.2.1.conv.bn.bias",
            "_head._modules_list.14.reg_convs.2.1.conv.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.2.1.conv.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.2.1.conv.bn.running_mean": "_head._modules_list.14.reg_convs.2.1.conv.bn.running_mean",
            "_head._modules_list.14.reg_convs.2.1.conv.bn.running_var": "_head._modules_list.14.reg_convs.2.1.conv.bn.running_var",
            "_head._modules_list.14.reg_convs.2.1.conv.bn.weight": "_head._modules_list.14.reg_convs.2.1.conv.bn.weight",
            "_head._modules_list.14.reg_convs.2.1.conv.conv.weight": "_head._modules_list.14.reg_convs.2.1.conv.conv.weight",
            "_head._modules_list.14.reg_convs.2.1.conv.weight": "_head._modules_list.14.reg_convs.2.1.conv.weight",
            "_head._modules_list.14.reg_convs.2.1.dconv.bn.bias": "_head._modules_list.14.reg_convs.2.1.dconv.bn.bias",
            "_head._modules_list.14.reg_convs.2.1.dconv.bn.num_batches_tracked": "_head._modules_list.14.reg_convs.2.1.dconv.bn.num_batches_tracked",
            "_head._modules_list.14.reg_convs.2.1.dconv.bn.running_mean": "_head._modules_list.14.reg_convs.2.1.dconv.bn.running_mean",
            "_head._modules_list.14.reg_convs.2.1.dconv.bn.running_var": "_head._modules_list.14.reg_convs.2.1.dconv.bn.running_var",
            "_head._modules_list.14.reg_convs.2.1.dconv.bn.weight": "_head._modules_list.14.reg_convs.2.1.dconv.bn.weight",
            "_head._modules_list.14.reg_convs.2.1.dconv.conv.weight": "_head._modules_list.14.reg_convs.2.1.dconv.conv.weight",
            "_head._modules_list.14.reg_preds.0.bias": "_head._modules_list.14.reg_preds.0.bias",
            "_head._modules_list.14.reg_preds.0.weight": "_head._modules_list.14.reg_preds.0.weight",
            "_head._modules_list.14.reg_preds.1.bias": "_head._modules_list.14.reg_preds.1.bias",
            "_head._modules_list.14.reg_preds.1.weight": "_head._modules_list.14.reg_preds.1.weight",
            "_head._modules_list.14.reg_preds.2.bias": "_head._modules_list.14.reg_preds.2.bias",
            "_head._modules_list.14.reg_preds.2.weight": "_head._modules_list.14.reg_preds.2.weight",
            "_head._modules_list.14.stems.0.bn.bias": "_head._modules_list.14.stems.0.bn.bias",
            "_head._modules_list.14.stems.0.bn.num_batches_tracked": "_head._modules_list.14.stems.0.bn.num_batches_tracked",
            "_head._modules_list.14.stems.0.bn.running_mean": "_head._modules_list.14.stems.0.bn.running_mean",
            "_head._modules_list.14.stems.0.bn.running_var": "_head._modules_list.14.stems.0.bn.running_var",
            "_head._modules_list.14.stems.0.bn.weight": "_head._modules_list.14.stems.0.bn.weight",
            "_head._modules_list.14.stems.0.conv.weight": "_head._modules_list.14.stems.0.conv.weight",
            "_head._modules_list.14.stems.1.bn.bias": "_head._modules_list.14.stems.1.bn.bias",
            "_head._modules_list.14.stems.1.bn.num_batches_tracked": "_head._modules_list.14.stems.1.bn.num_batches_tracked",
            "_head._modules_list.14.stems.1.bn.running_mean": "_head._modules_list.14.stems.1.bn.running_mean",
            "_head._modules_list.14.stems.1.bn.running_var": "_head._modules_list.14.stems.1.bn.running_var",
            "_head._modules_list.14.stems.1.bn.weight": "_head._modules_list.14.stems.1.bn.weight",
            "_head._modules_list.14.stems.1.conv.weight": "_head._modules_list.14.stems.1.conv.weight",
            "_head._modules_list.14.stems.2.bn.bias": "_head._modules_list.14.stems.2.bn.bias",
            "_head._modules_list.14.stems.2.bn.num_batches_tracked": "_head._modules_list.14.stems.2.bn.num_batches_tracked",
            "_head._modules_list.14.stems.2.bn.running_mean": "_head._modules_list.14.stems.2.bn.running_mean",
            "_head._modules_list.14.stems.2.bn.running_var": "_head._modules_list.14.stems.2.bn.running_var",
            "_head._modules_list.14.stems.2.bn.weight": "_head._modules_list.14.stems.2.bn.weight",
            "_head._modules_list.14.stems.2.conv.weight": "_head._modules_list.14.stems.2.conv.weight",
            "_head._modules_list.3.cv1.bn.bias": "_head._modules_list.3.conv1.bn.bias",
            "_head._modules_list.3.cv1.bn.num_batches_tracked": "_head._modules_list.3.conv1.bn.num_batches_tracked",
            "_head._modules_list.3.cv1.bn.running_mean": "_head._modules_list.3.conv1.bn.running_mean",
            "_head._modules_list.3.cv1.bn.running_var": "_head._modules_list.3.conv1.bn.running_var",
            "_head._modules_list.3.cv1.bn.weight": "_head._modules_list.3.conv1.bn.weight",
            "_head._modules_list.3.cv1.conv.weight": "_head._modules_list.3.conv1.conv.weight",
            "_head._modules_list.3.cv2.bn.bias": "_head._modules_list.3.conv2.bn.bias",
            "_head._modules_list.3.cv2.bn.num_batches_tracked": "_head._modules_list.3.conv2.bn.num_batches_tracked",
            "_head._modules_list.3.cv2.bn.running_mean": "_head._modules_list.3.conv2.bn.running_mean",
            "_head._modules_list.3.cv2.bn.running_var": "_head._modules_list.3.conv2.bn.running_var",
            "_head._modules_list.3.cv2.bn.weight": "_head._modules_list.3.conv2.bn.weight",
            "_head._modules_list.3.cv2.conv.weight": "_head._modules_list.3.conv2.conv.weight",
            "_head._modules_list.3.cv3.bn.bias": "_head._modules_list.3.conv3.bn.bias",
            "_head._modules_list.3.cv3.bn.num_batches_tracked": "_head._modules_list.3.conv3.bn.num_batches_tracked",
            "_head._modules_list.3.cv3.bn.running_mean": "_head._modules_list.3.conv3.bn.running_mean",
            "_head._modules_list.3.cv3.bn.running_var": "_head._modules_list.3.conv3.bn.running_var",
            "_head._modules_list.3.cv3.bn.weight": "_head._modules_list.3.conv3.bn.weight",
            "_head._modules_list.3.cv3.conv.weight": "_head._modules_list.3.conv3.conv.weight",
            "_head._modules_list.3.m.0.cv1.bn.bias": "_head._modules_list.3.bottlenecks.0.cv1.bn.bias",
            "_head._modules_list.3.m.0.cv1.bn.num_batches_tracked": "_head._modules_list.3.bottlenecks.0.cv1.bn.num_batches_tracked",
            "_head._modules_list.3.m.0.cv1.bn.running_mean": "_head._modules_list.3.bottlenecks.0.cv1.bn.running_mean",
            "_head._modules_list.3.m.0.cv1.bn.running_var": "_head._modules_list.3.bottlenecks.0.cv1.bn.running_var",
            "_head._modules_list.3.m.0.cv1.bn.weight": "_head._modules_list.3.bottlenecks.0.cv1.bn.weight",
            "_head._modules_list.3.m.0.cv1.conv.weight": "_head._modules_list.3.bottlenecks.0.cv1.conv.weight",
            "_head._modules_list.3.m.0.cv2.bn.bias": "_head._modules_list.3.bottlenecks.0.cv2.bn.bias",
            "_head._modules_list.3.m.0.cv2.bn.num_batches_tracked": "_head._modules_list.3.bottlenecks.0.cv2.bn.num_batches_tracked",
            "_head._modules_list.3.m.0.cv2.bn.running_mean": "_head._modules_list.3.bottlenecks.0.cv2.bn.running_mean",
            "_head._modules_list.3.m.0.cv2.bn.running_var": "_head._modules_list.3.bottlenecks.0.cv2.bn.running_var",
            "_head._modules_list.3.m.0.cv2.bn.weight": "_head._modules_list.3.bottlenecks.0.cv2.bn.weight",
            "_head._modules_list.3.m.0.cv2.conv.bn.bias": "_head._modules_list.3.bottlenecks.0.cv2.conv.bn.bias",
            "_head._modules_list.3.m.0.cv2.conv.bn.num_batches_tracked": "_head._modules_list.3.bottlenecks.0.cv2.conv.bn.num_batches_tracked",
            "_head._modules_list.3.m.0.cv2.conv.bn.running_mean": "_head._modules_list.3.bottlenecks.0.cv2.conv.bn.running_mean",
            "_head._modules_list.3.m.0.cv2.conv.bn.running_var": "_head._modules_list.3.bottlenecks.0.cv2.conv.bn.running_var",
            "_head._modules_list.3.m.0.cv2.conv.bn.weight": "_head._modules_list.3.bottlenecks.0.cv2.conv.bn.weight",
            "_head._modules_list.3.m.0.cv2.conv.conv.weight": "_head._modules_list.3.bottlenecks.0.cv2.conv.conv.weight",
            "_head._modules_list.3.m.0.cv2.conv.weight": "_head._modules_list.3.bottlenecks.0.cv2.conv.weight",
            "_head._modules_list.3.m.0.cv2.dconv.bn.bias": "_head._modules_list.3.bottlenecks.0.cv2.dconv.bn.bias",
            "_head._modules_list.3.m.0.cv2.dconv.bn.num_batches_tracked": "_head._modules_list.3.bottlenecks.0.cv2.dconv.bn.num_batches_tracked",
            "_head._modules_list.3.m.0.cv2.dconv.bn.running_mean": "_head._modules_list.3.bottlenecks.0.cv2.dconv.bn.running_mean",
            "_head._modules_list.3.m.0.cv2.dconv.bn.running_var": "_head._modules_list.3.bottlenecks.0.cv2.dconv.bn.running_var",
            "_head._modules_list.3.m.0.cv2.dconv.bn.weight": "_head._modules_list.3.bottlenecks.0.cv2.dconv.bn.weight",
            "_head._modules_list.3.m.0.cv2.dconv.conv.weight": "_head._modules_list.3.bottlenecks.0.cv2.dconv.conv.weight",
            "_head._modules_list.3.m.1.cv1.bn.bias": "_head._modules_list.3.bottlenecks.1.cv1.bn.bias",
            "_head._modules_list.3.m.1.cv1.bn.num_batches_tracked": "_head._modules_list.3.bottlenecks.1.cv1.bn.num_batches_tracked",
            "_head._modules_list.3.m.1.cv1.bn.running_mean": "_head._modules_list.3.bottlenecks.1.cv1.bn.running_mean",
            "_head._modules_list.3.m.1.cv1.bn.running_var": "_head._modules_list.3.bottlenecks.1.cv1.bn.running_var",
            "_head._modules_list.3.m.1.cv1.bn.weight": "_head._modules_list.3.bottlenecks.1.cv1.bn.weight",
            "_head._modules_list.3.m.1.cv1.conv.weight": "_head._modules_list.3.bottlenecks.1.cv1.conv.weight",
            "_head._modules_list.3.m.1.cv2.bn.bias": "_head._modules_list.3.bottlenecks.1.cv2.bn.bias",
            "_head._modules_list.3.m.1.cv2.bn.num_batches_tracked": "_head._modules_list.3.bottlenecks.1.cv2.bn.num_batches_tracked",
            "_head._modules_list.3.m.1.cv2.bn.running_mean": "_head._modules_list.3.bottlenecks.1.cv2.bn.running_mean",
            "_head._modules_list.3.m.1.cv2.bn.running_var": "_head._modules_list.3.bottlenecks.1.cv2.bn.running_var",
            "_head._modules_list.3.m.1.cv2.bn.weight": "_head._modules_list.3.bottlenecks.1.cv2.bn.weight",
            "_head._modules_list.3.m.1.cv2.conv.weight": "_head._modules_list.3.bottlenecks.1.cv2.conv.weight",
            "_head._modules_list.3.m.2.cv1.bn.bias": "_head._modules_list.3.bottlenecks.2.cv1.bn.bias",
            "_head._modules_list.3.m.2.cv1.bn.num_batches_tracked": "_head._modules_list.3.bottlenecks.2.cv1.bn.num_batches_tracked",
            "_head._modules_list.3.m.2.cv1.bn.running_mean": "_head._modules_list.3.bottlenecks.2.cv1.bn.running_mean",
            "_head._modules_list.3.m.2.cv1.bn.running_var": "_head._modules_list.3.bottlenecks.2.cv1.bn.running_var",
            "_head._modules_list.3.m.2.cv1.bn.weight": "_head._modules_list.3.bottlenecks.2.cv1.bn.weight",
            "_head._modules_list.3.m.2.cv1.conv.weight": "_head._modules_list.3.bottlenecks.2.cv1.conv.weight",
            "_head._modules_list.3.m.2.cv2.bn.bias": "_head._modules_list.3.bottlenecks.2.cv2.bn.bias",
            "_head._modules_list.3.m.2.cv2.bn.num_batches_tracked": "_head._modules_list.3.bottlenecks.2.cv2.bn.num_batches_tracked",
            "_head._modules_list.3.m.2.cv2.bn.running_mean": "_head._modules_list.3.bottlenecks.2.cv2.bn.running_mean",
            "_head._modules_list.3.m.2.cv2.bn.running_var": "_head._modules_list.3.bottlenecks.2.cv2.bn.running_var",
            "_head._modules_list.3.m.2.cv2.bn.weight": "_head._modules_list.3.bottlenecks.2.cv2.bn.weight",
            "_head._modules_list.3.m.2.cv2.conv.weight": "_head._modules_list.3.bottlenecks.2.cv2.conv.weight",
            "_head._modules_list.4.bn.bias": "_head._modules_list.4.bn.bias",
            "_head._modules_list.4.bn.num_batches_tracked": "_head._modules_list.4.bn.num_batches_tracked",
            "_head._modules_list.4.bn.running_mean": "_head._modules_list.4.bn.running_mean",
            "_head._modules_list.4.bn.running_var": "_head._modules_list.4.bn.running_var",
            "_head._modules_list.4.bn.weight": "_head._modules_list.4.bn.weight",
            "_head._modules_list.4.conv.weight": "_head._modules_list.4.conv.weight",
            "_head._modules_list.7.cv1.bn.bias": "_head._modules_list.7.conv1.bn.bias",
            "_head._modules_list.7.cv1.bn.num_batches_tracked": "_head._modules_list.7.conv1.bn.num_batches_tracked",
            "_head._modules_list.7.cv1.bn.running_mean": "_head._modules_list.7.conv1.bn.running_mean",
            "_head._modules_list.7.cv1.bn.running_var": "_head._modules_list.7.conv1.bn.running_var",
            "_head._modules_list.7.cv1.bn.weight": "_head._modules_list.7.conv1.bn.weight",
            "_head._modules_list.7.cv1.conv.weight": "_head._modules_list.7.conv1.conv.weight",
            "_head._modules_list.7.cv2.bn.bias": "_head._modules_list.7.conv2.bn.bias",
            "_head._modules_list.7.cv2.bn.num_batches_tracked": "_head._modules_list.7.conv2.bn.num_batches_tracked",
            "_head._modules_list.7.cv2.bn.running_mean": "_head._modules_list.7.conv2.bn.running_mean",
            "_head._modules_list.7.cv2.bn.running_var": "_head._modules_list.7.conv2.bn.running_var",
            "_head._modules_list.7.cv2.bn.weight": "_head._modules_list.7.conv2.bn.weight",
            "_head._modules_list.7.cv2.conv.weight": "_head._modules_list.7.conv2.conv.weight",
            "_head._modules_list.7.cv3.bn.bias": "_head._modules_list.7.conv3.bn.bias",
            "_head._modules_list.7.cv3.bn.num_batches_tracked": "_head._modules_list.7.conv3.bn.num_batches_tracked",
            "_head._modules_list.7.cv3.bn.running_mean": "_head._modules_list.7.conv3.bn.running_mean",
            "_head._modules_list.7.cv3.bn.running_var": "_head._modules_list.7.conv3.bn.running_var",
            "_head._modules_list.7.cv3.bn.weight": "_head._modules_list.7.conv3.bn.weight",
            "_head._modules_list.7.cv3.conv.weight": "_head._modules_list.7.conv3.conv.weight",
            "_head._modules_list.7.m.0.cv1.bn.bias": "_head._modules_list.7.bottlenecks.0.cv1.bn.bias",
            "_head._modules_list.7.m.0.cv1.bn.num_batches_tracked": "_head._modules_list.7.bottlenecks.0.cv1.bn.num_batches_tracked",
            "_head._modules_list.7.m.0.cv1.bn.running_mean": "_head._modules_list.7.bottlenecks.0.cv1.bn.running_mean",
            "_head._modules_list.7.m.0.cv1.bn.running_var": "_head._modules_list.7.bottlenecks.0.cv1.bn.running_var",
            "_head._modules_list.7.m.0.cv1.bn.weight": "_head._modules_list.7.bottlenecks.0.cv1.bn.weight",
            "_head._modules_list.7.m.0.cv1.conv.weight": "_head._modules_list.7.bottlenecks.0.cv1.conv.weight",
            "_head._modules_list.7.m.0.cv2.bn.bias": "_head._modules_list.7.bottlenecks.0.cv2.bn.bias",
            "_head._modules_list.7.m.0.cv2.bn.num_batches_tracked": "_head._modules_list.7.bottlenecks.0.cv2.bn.num_batches_tracked",
            "_head._modules_list.7.m.0.cv2.bn.running_mean": "_head._modules_list.7.bottlenecks.0.cv2.bn.running_mean",
            "_head._modules_list.7.m.0.cv2.bn.running_var": "_head._modules_list.7.bottlenecks.0.cv2.bn.running_var",
            "_head._modules_list.7.m.0.cv2.bn.weight": "_head._modules_list.7.bottlenecks.0.cv2.bn.weight",
            "_head._modules_list.7.m.0.cv2.conv.bn.bias": "_head._modules_list.7.bottlenecks.0.cv2.conv.bn.bias",
            "_head._modules_list.7.m.0.cv2.conv.bn.num_batches_tracked": "_head._modules_list.7.bottlenecks.0.cv2.conv.bn.num_batches_tracked",
            "_head._modules_list.7.m.0.cv2.conv.bn.running_mean": "_head._modules_list.7.bottlenecks.0.cv2.conv.bn.running_mean",
            "_head._modules_list.7.m.0.cv2.conv.bn.running_var": "_head._modules_list.7.bottlenecks.0.cv2.conv.bn.running_var",
            "_head._modules_list.7.m.0.cv2.conv.bn.weight": "_head._modules_list.7.bottlenecks.0.cv2.conv.bn.weight",
            "_head._modules_list.7.m.0.cv2.conv.conv.weight": "_head._modules_list.7.bottlenecks.0.cv2.conv.conv.weight",
            "_head._modules_list.7.m.0.cv2.conv.weight": "_head._modules_list.7.bottlenecks.0.cv2.conv.weight",
            "_head._modules_list.7.m.0.cv2.dconv.bn.bias": "_head._modules_list.7.bottlenecks.0.cv2.dconv.bn.bias",
            "_head._modules_list.7.m.0.cv2.dconv.bn.num_batches_tracked": "_head._modules_list.7.bottlenecks.0.cv2.dconv.bn.num_batches_tracked",
            "_head._modules_list.7.m.0.cv2.dconv.bn.running_mean": "_head._modules_list.7.bottlenecks.0.cv2.dconv.bn.running_mean",
            "_head._modules_list.7.m.0.cv2.dconv.bn.running_var": "_head._modules_list.7.bottlenecks.0.cv2.dconv.bn.running_var",
            "_head._modules_list.7.m.0.cv2.dconv.bn.weight": "_head._modules_list.7.bottlenecks.0.cv2.dconv.bn.weight",
            "_head._modules_list.7.m.0.cv2.dconv.conv.weight": "_head._modules_list.7.bottlenecks.0.cv2.dconv.conv.weight",
            "_head._modules_list.7.m.1.cv1.bn.bias": "_head._modules_list.7.bottlenecks.1.cv1.bn.bias",
            "_head._modules_list.7.m.1.cv1.bn.num_batches_tracked": "_head._modules_list.7.bottlenecks.1.cv1.bn.num_batches_tracked",
            "_head._modules_list.7.m.1.cv1.bn.running_mean": "_head._modules_list.7.bottlenecks.1.cv1.bn.running_mean",
            "_head._modules_list.7.m.1.cv1.bn.running_var": "_head._modules_list.7.bottlenecks.1.cv1.bn.running_var",
            "_head._modules_list.7.m.1.cv1.bn.weight": "_head._modules_list.7.bottlenecks.1.cv1.bn.weight",
            "_head._modules_list.7.m.1.cv1.conv.weight": "_head._modules_list.7.bottlenecks.1.cv1.conv.weight",
            "_head._modules_list.7.m.1.cv2.bn.bias": "_head._modules_list.7.bottlenecks.1.cv2.bn.bias",
            "_head._modules_list.7.m.1.cv2.bn.num_batches_tracked": "_head._modules_list.7.bottlenecks.1.cv2.bn.num_batches_tracked",
            "_head._modules_list.7.m.1.cv2.bn.running_mean": "_head._modules_list.7.bottlenecks.1.cv2.bn.running_mean",
            "_head._modules_list.7.m.1.cv2.bn.running_var": "_head._modules_list.7.bottlenecks.1.cv2.bn.running_var",
            "_head._modules_list.7.m.1.cv2.bn.weight": "_head._modules_list.7.bottlenecks.1.cv2.bn.weight",
            "_head._modules_list.7.m.1.cv2.conv.weight": "_head._modules_list.7.bottlenecks.1.cv2.conv.weight",
            "_head._modules_list.7.m.2.cv1.bn.bias": "_head._modules_list.7.bottlenecks.2.cv1.bn.bias",
            "_head._modules_list.7.m.2.cv1.bn.num_batches_tracked": "_head._modules_list.7.bottlenecks.2.cv1.bn.num_batches_tracked",
            "_head._modules_list.7.m.2.cv1.bn.running_mean": "_head._modules_list.7.bottlenecks.2.cv1.bn.running_mean",
            "_head._modules_list.7.m.2.cv1.bn.running_var": "_head._modules_list.7.bottlenecks.2.cv1.bn.running_var",
            "_head._modules_list.7.m.2.cv1.bn.weight": "_head._modules_list.7.bottlenecks.2.cv1.bn.weight",
            "_head._modules_list.7.m.2.cv1.conv.weight": "_head._modules_list.7.bottlenecks.2.cv1.conv.weight",
            "_head._modules_list.7.m.2.cv2.bn.bias": "_head._modules_list.7.bottlenecks.2.cv2.bn.bias",
            "_head._modules_list.7.m.2.cv2.bn.num_batches_tracked": "_head._modules_list.7.bottlenecks.2.cv2.bn.num_batches_tracked",
            "_head._modules_list.7.m.2.cv2.bn.running_mean": "_head._modules_list.7.bottlenecks.2.cv2.bn.running_mean",
            "_head._modules_list.7.m.2.cv2.bn.running_var": "_head._modules_list.7.bottlenecks.2.cv2.bn.running_var",
            "_head._modules_list.7.m.2.cv2.bn.weight": "_head._modules_list.7.bottlenecks.2.cv2.bn.weight",
            "_head._modules_list.7.m.2.cv2.conv.weight": "_head._modules_list.7.bottlenecks.2.cv2.conv.weight",
            "_head._modules_list.8.bn.bias": "_head._modules_list.8.bn.bias",
            "_head._modules_list.8.bn.num_batches_tracked": "_head._modules_list.8.bn.num_batches_tracked",
            "_head._modules_list.8.bn.running_mean": "_head._modules_list.8.bn.running_mean",
            "_head._modules_list.8.bn.running_var": "_head._modules_list.8.bn.running_var",
            "_head._modules_list.8.bn.weight": "_head._modules_list.8.bn.weight",
            "_head._modules_list.8.conv.bn.bias": "_head._modules_list.8.conv.bn.bias",
            "_head._modules_list.8.conv.bn.num_batches_tracked": "_head._modules_list.8.conv.bn.num_batches_tracked",
            "_head._modules_list.8.conv.bn.running_mean": "_head._modules_list.8.conv.bn.running_mean",
            "_head._modules_list.8.conv.bn.running_var": "_head._modules_list.8.conv.bn.running_var",
            "_head._modules_list.8.conv.bn.weight": "_head._modules_list.8.conv.bn.weight",
            "_head._modules_list.8.conv.conv.weight": "_head._modules_list.8.conv.conv.weight",
            "_head._modules_list.8.conv.weight": "_head._modules_list.8.conv.weight",
            "_head._modules_list.8.dconv.bn.bias": "_head._modules_list.8.dconv.bn.bias",
            "_head._modules_list.8.dconv.bn.num_batches_tracked": "_head._modules_list.8.dconv.bn.num_batches_tracked",
            "_head._modules_list.8.dconv.bn.running_mean": "_head._modules_list.8.dconv.bn.running_mean",
            "_head._modules_list.8.dconv.bn.running_var": "_head._modules_list.8.dconv.bn.running_var",
            "_head._modules_list.8.dconv.bn.weight": "_head._modules_list.8.dconv.bn.weight",
            "_head._modules_list.8.dconv.conv.weight": "_head._modules_list.8.dconv.conv.weight",
        }

    def __call__(self, model_state_dict: Mapping[str, Tensor], checkpoint_state_dict: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        checkpoint_state_dict = self._remove_saved_stride_tensors(checkpoint_state_dict)
        checkpoint_state_dict = self._reshape_old_focus_weights(checkpoint_state_dict)
        checkpoint_state_dict = self._rename_layers(checkpoint_state_dict)
        return checkpoint_state_dict

    def _remove_saved_stride_tensors(self, state_dict):
        exclude_stride_keys = {
            "stride",
            "_head.anchors._anchors",
            "_head.anchors._anchor_grid",
            "_head.anchors._stride",
            "_head._modules_list.14.stride",
        }
        return collections.OrderedDict([(k, v) for k, v in state_dict.items() if k not in exclude_stride_keys])

    def _rename_layers(self, state_dict):
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            k = self.layers_rename_table.get(k, k)
            new_state_dict[k] = v
        return new_state_dict

    def _reshape_old_focus_weights(self, state_dict):
        if "_backbone._modules_list.0.conv.conv.weight" in state_dict:
            layer = state_dict["_backbone._modules_list.0.conv.conv.weight"]
            del state_dict["_backbone._modules_list.0.conv.conv.weight"]

            data = torch.zeros((layer.size(0), 3, 6, 6))
            data[:, :, ::2, ::2] = layer.data[:, :3]
            data[:, :, 1::2, ::2] = layer.data[:, 3:6]
            data[:, :, ::2, 1::2] = layer.data[:, 6:9]
            data[:, :, 1::2, 1::2] = layer.data[:, 9:12]
            state_dict["_backbone._modules_list.0.conv.weight"] = data

        return state_dict

    def _yolox_ckpt_solver(self, ckpt_key, ckpt_val, model_key, model_val):
        """
        Helper method for reshaping old pretrained checkpoint's focus weights to 6x6 conv weights.
        """

        if (
            ckpt_val.shape != model_val.shape
            and (ckpt_key == "module._backbone._modules_list.0.conv.conv.weight" or ckpt_key == "_backbone._modules_list.0.conv.conv.weight")
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
    if solver is None:
        solver = DefaultCheckpointSolver()

    if "net" in source_ckpt.keys():
        source_ckpt = source_ckpt["net"]

    if len(exclude):
        model_state_dict = {k: v for k, v in model_state_dict.items() if not any(x in k for x in exclude)}

    new_ckpt_dict = solver(model_state_dict, source_ckpt)
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


@resolve_param("strict", TypeFactory.from_enum_cls(StrictLoad))
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

    :param net:                    Network to load the checkpoint to
    :param ckpt_local_path:        Local path to the checkpoint file
    :param load_ema_as_net:        Will load the EMA inside the checkpoint file to the network when set
    :param load_backbone:          Whether to load the checkpoint as a backbone
    :param strict:                 See super_gradients.common.data_types.enum.strict_load.StrictLoad class documentation for details
                                   (default=NO_KEY_MATCHING to suport SG trained checkpoints)
    :param load_weights_only:      Whether to ignore all other entries other then "net".
    :param load_processing_params: Whether to call set_dataset_processing_params on "processing_params" entry inside the
                                   checkpoint file (default=False).
    :return:
    """
    net = unwrap_model(net)

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

    _maybe_load_preprocessing_params(net, checkpoint)

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


def load_pretrained_weights(model: torch.nn.Module, architecture: str, pretrained_weights: str):
    """
    Loads pretrained weights from the MODEL_URLS dictionary to model

    :param architecture:        name of the model's architecture
    :param model:               model to load pretrinaed weights for
    :param pretrained_weights:  name for the pretrianed weights (i.e imagenet)

    :return:                    None
    """
    from super_gradients.common.object_names import Models

    model_url_key = architecture + "_" + str(pretrained_weights)
    if model_url_key not in MODEL_URLS.keys():
        raise MissingPretrainedWeightsException(model_url_key)

    if pretrained_weights in DATASET_LICENSES:
        logger.warning(
            f":warning: The pre-trained models provided by SuperGradients may have their own licenses or terms and "
            "conditions derived from the dataset used for pre-training.\n It is your responsibility to determine whether you "
            "have permission to use the models for your use case.\n The model you have requested was pre-trained on the "
            f"{pretrained_weights} dataset, published under the following terms: {DATASET_LICENSES[pretrained_weights]}"
        )
    url = MODEL_URLS[model_url_key]

    if architecture in {Models.YOLO_NAS_S, Models.YOLO_NAS_M, Models.YOLO_NAS_L}:
        logger.info(
            "License Notification: YOLO-NAS pre-trained weights are subjected to the specific license terms and conditions detailed in \n"
            "https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md\n"
            "By downloading the pre-trained weight files you agree to comply with these terms."
        )
    elif architecture in {Models.YOLO_NAS_POSE_N, Models.YOLO_NAS_POSE_S, Models.YOLO_NAS_POSE_M, Models.YOLO_NAS_POSE_L}:
        logger.info(
            "License Notification: YOLO-NAS-POSE pre-trained weights are subjected to the specific license terms and conditions detailed in \n"
            "https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS-POSE.md\n"
            "By downloading the pre-trained weight files you agree to comply with these terms."
        )

    # Basically this check allows settings pretrained weights from local path using file:///path/to/weights scheme
    # which is a valid URI scheme for local files
    # Supporting local files and file URI allows us modification of pretrained weights dics in unit tests
    if url.startswith("file://") or os.path.exists(url):
        pretrained_state_dict = torch.load(url.replace("file://", ""), map_location="cpu")
    else:
        unique_filename = url.split("https://sg-hub-nv.s3.amazonaws.com/models/")[1].replace("/", "_").replace(" ", "_")
        map_location = torch.device("cpu")
        with wait_for_the_master(get_local_rank()):
            pretrained_state_dict = load_state_dict_from_url(url=url, map_location=map_location, file_name=unique_filename)

    _load_weights(architecture, model, pretrained_state_dict)
    _maybe_load_preprocessing_params(model, pretrained_state_dict)


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
    _maybe_load_preprocessing_params(model, pretrained_state_dict)


def _load_weights(architecture, model, pretrained_state_dict):
    if "ema_net" in pretrained_state_dict.keys():
        pretrained_state_dict["net"] = pretrained_state_dict["ema_net"]
    solver = YoloXCheckpointSolver() if "yolox" in architecture else DefaultCheckpointSolver()
    adaptive_load_state_dict(net=model, state_dict=pretrained_state_dict, strict=StrictLoad.NO_KEY_MATCHING, solver=solver)
    logger.info(f"Successfully loaded pretrained weights for architecture {architecture}")


def _maybe_load_preprocessing_params(model: Union[nn.Module, HasPredict], checkpoint: Mapping[str, Tensor]) -> bool:
    """
    Tries to load preprocessing params from the checkpoint to the model.
    The function does not crash, and raises a warning if the loading fails.
    :param model:      Instance of nn.Module
    :param checkpoint: Entire checkpoint dict (not state_dict with model weights)
    :return:           True if the loading was successful, False otherwise.
    """
    model = unwrap_model(model)
    checkpoint_has_preprocessing_params = "processing_params" in checkpoint.keys()
    model_has_predict = isinstance(model, HasPredict)
    logger.debug(
        f"Trying to load preprocessing params from checkpoint. Preprocessing params in checkpoint: {checkpoint_has_preprocessing_params}. "
        f"Model {model.__class__.__name__} inherit HasPredict: {model_has_predict}"
    )

    if model_has_predict and checkpoint_has_preprocessing_params:
        try:
            model.set_dataset_processing_params(**checkpoint["processing_params"])
            logger.debug(f"Successfully loaded preprocessing params from checkpoint {checkpoint['processing_params']}")
            return True
        except Exception as e:
            logger.warning(
                f"Could not set preprocessing pipeline from the checkpoint dataset: {e}. Before calling"
                "predict make sure to call set_dataset_processing_params."
            )
    return False


def get_scheduler_state(scheduler) -> Dict[str, Tensor]:
    """
    Wrapper for getting a torch lr scheduler state dict, resolving some issues with CyclicLR
    (see https://github.com/pytorch/pytorch/pull/91400)
    :param scheduler: torch.optim.lr_scheduler._LRScheduler, the scheduler
    :return:          the scheduler's state_dict
    """
    from super_gradients.training.utils import torch_version_is_greater_or_equal
    from torch.optim.lr_scheduler import CyclicLR

    state = scheduler.state_dict()
    if isinstance(scheduler, CyclicLR) and not torch_version_is_greater_or_equal(2, 0):
        # A check is needed since torch 1.12 does not have the _scale_fn_ref attribute, while other versions do
        if "_scale_fn_ref" in state:
            del state["_scale_fn_ref"]
    return state
