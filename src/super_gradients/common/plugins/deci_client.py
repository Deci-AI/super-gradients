import os
import json
import sys
import shutil
import urllib
from zipfile import ZipFile
import socket
import urllib.error
from urllib.request import urlretrieve
from typing import List, Optional, Sequence

import importlib.util

import torch.hub
from omegaconf import DictConfig
from torch import nn

import super_gradients
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.cfg_utils import load_arch_params, load_recipe
from super_gradients.common.environment.path_utils import normalize_path

logger = get_logger(__name__)

client_enabled = True
try:
    from deci_platform_client import DeciPlatformClient
    from deci_platform_client.types import S3SignedUrl
    from deci_platform_client.models import (
        ModelBenchmarkState,
        SentryLevel,
        FrameworkType,
        HardwareType,
        QuantizationLevel,
    )
    from deci_common.data_interfaces.files_data_interface import FileDownloadFailedException
    from deci_platform_client.models import AutoNACFileName
    from deci_platform_client.exceptions import ApiException, ApiTypeError

except (ImportError, NameError):
    client_enabled = False

DOWNLOAD_MODEL_TIMEOUT_SECONDS = 5 * 60


class DeciClient:
    """
    A client to deci platform and model zoo.
    requires credentials for connection
    """

    def __init__(self):
        if not client_enabled:
            logger.error(
                "deci-platform-client or deci-common are not installed. Model cannot be loaded from deci lab."
                "Please install deci-platform-client>=5.0.0 and deci-common>=12.0.0"
            )
            return

        self.lab_client = DeciPlatformClient()

    def _get_file(self, model_name: str, file_name: "AutoNACFileName") -> Optional[str]:
        """Get a file from the DeciPlatform if it exists, otherwise returns None
        :param model_name:      Name of the model to download from, as saved in the platform.
        :param file_name:       Name of the file to download
        :return:            Path were the downloaded file was saved to. None if not found.
        """
        try:
            download_link, etag = self.lab_client.get_autonac_model_file_link(
                model_name=model_name,
                file_name=file_name,
                super_gradients_version=super_gradients.__version__,
            )
        except ApiException as e:
            if e.status == 401:
                logger.error(
                    "Unauthorized. wrong credentials or credentials not defined. "
                    "Please provide credentials via environment variables (DECI_CLIENT_ID, DECI_CLIENT_SECRET)"
                )
            elif e.status == 400 and e.body is not None and "message" in e.body:
                logger.error(f"Deci client: {json.loads(e.body)['message']}")
            else:
                logger.debug(e.body)
            return None
        cache_dir = os.path.join(torch.hub.get_dir(), "deci")
        file_path = os.path.join(cache_dir, etag or "", os.path.basename(file_name))
        file_path = normalize_path(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.isfile(file_path):
            return file_path

        file_path = self._download_file_to_cache_dir(
            file_url=download_link,
            file_path=file_path,
        )
        return file_path

    def _download_file_to_cache_dir(self, file_url: str, file_path: str, timeout_seconds: Optional[int] = DOWNLOAD_MODEL_TIMEOUT_SECONDS):
        """
        Download a file from a url to a cache dir. The file will be saved in a subfolder named by the etag.
        This allow us to save multiple versions of the same file and cache them, so when a file with the same etag is
        requested, we can return the cached file.

        :param file_url:  Url to download the file from.
        :param file_path: Path to save the file to.
        :return:        Path were the downloaded file was saved to. (same as file_path)
        """
        # TODO: Use requests with stream and limit the file size and timeouts.
        socket.setdefaulttimeout(timeout_seconds)
        try:
            urlretrieve(file_url, file_path)
        except urllib.error.ContentTooShortError as ex:
            raise FileDownloadFailedException("File download did not finish correctly " + str(ex))
        return file_path

    def get_model_arch_params(self, model_name: str) -> Optional[DictConfig]:
        """Get the model arch_params from DeciPlatform.
        :param model_name:  Name of the model as saved in the platform.
        :return:            arch_params. None if arch_params were not found for this specific model on this SG version."""
        arch_params_file = self._get_file(model_name, AutoNACFileName.STRUCTURE_YAML)
        if arch_params_file is None:
            return None

        config_name = os.path.basename(arch_params_file)
        download_dir = os.path.dirname(arch_params_file)

        # The arch_params config files need to be saved inside an "arch_params" folder
        _move_file_to_folder(src_file_path=arch_params_file, dest_dir_name="arch_params")

        return load_arch_params(config_name=config_name, recipes_dir_path=download_dir)

    def get_model_recipe(self, model_name: str) -> Optional[DictConfig]:
        """Get the model recipe from DeciPlatform.
        :param model_name:  Name of the model as saved in the platform.
        :return:            recipe. None if recipe were not found for this specific model on this SG version."""
        recipe_file = self._get_file(model_name, AutoNACFileName.RECIPE_YAML)
        if recipe_file is None:
            return None

        config_name = os.path.basename(recipe_file)
        download_dir = os.path.dirname(recipe_file)

        return load_recipe(config_name=config_name, recipes_dir_path=download_dir)

    def get_model_weights(self, model_name: str) -> Optional[str]:
        """Get the path to model weights (downloaded locally).
        :param model_name:  Name of the model as saved in the platform.
        :return:            model_weights path. None if weights were not found for this specific model on this SG version."""
        return self._get_file(model_name=model_name, file_name=AutoNACFileName.WEIGHTS_PTH)

    def download_and_load_model_additional_code(self, model_name: str, target_path: str, package_name: str = "deci_model_code") -> None:
        """
        try to download code files for this model.
        if found, code files will be placed in the target_path/package_name and imported dynamically
        """

        file = self._get_file(model_name=model_name, file_name=AutoNACFileName.CODE_ZIP)

        package_path = os.path.join(target_path, package_name)
        if file is not None:
            # crete the directory
            os.makedirs(package_path, exist_ok=True)

            # extract code files
            with ZipFile(file) as zipfile:
                zipfile.extractall(package_path)

            # add an init file that imports all code files
            with open(os.path.join(package_path, "__init__.py"), "w") as init_file:
                all_str = "\n\n__all__ = ["
                for code_file in os.listdir(path=package_path):
                    if code_file.endswith(".py") and not code_file.startswith("__init__"):
                        init_file.write(f'import {code_file.replace(".py", "")}\n')
                        all_str += f'"{code_file.replace(".py", "")}", '

                all_str += "]\n\n"
                init_file.write(all_str)

            # include in path and import
            sys.path.insert(1, package_path)
            importlib.import_module(package_name)

            logger.info(
                f"*** IMPORTANT ***: files required for the model {model_name} were downloaded and added to your code in:\n{package_path}\n"
                f"These files will be downloaded to the same location each time the model is fetched from the deci-client.\n"
                f"you can override this by passing models.get(... download_required_code=False) and importing the files yourself"
            )

    def upload_model(
        self,
        model: nn.Module,
        name: str,
        input_dimensions: "Sequence[int]",
        target_hardware_types: "Optional[List[HardwareType]]" = None,
        target_quantization_level: "Optional[QuantizationLevel]" = None,
        target_batch_size: "Optional[int]" = None,
    ):
        """
        This function will upload the trained model to the Deci Lab

        :param model:                            The resulting model from the training process
        :param name:                             The model's name
        :param input_dimensions:                 The model's input dimensions
        :param target_hardware_types:            List of hardware types to optimize the model for
        :param target_quantization_level:        The quantization level to optimize the model for
        :param target_batch_size:                The batch size to optimize the model for
        """
        model_id = self.lab_client.register_model(
            model=model,
            name=name,
            framework=FrameworkType.PYTORCH,
            input_dimensions=input_dimensions,
        )
        if target_hardware_types:
            kwargs = {}
            if target_quantization_level:
                kwargs["quantization_level"] = target_quantization_level
            if target_batch_size:
                kwargs["batch_size"] = target_batch_size
            self.lab_client.optimize_model(model_id=model_id, hardware_types=target_hardware_types, **kwargs)

    def is_model_benchmarking(self, name: str) -> bool:
        """Check if a given model is still benchmarking or not.
        :param name: The mode name.
        """
        benchmark_state = self.lab_client.get_model(name=name)[0]["benchmarkState"]
        return benchmark_state in [ModelBenchmarkState.IN_PROGRESS, ModelBenchmarkState.PENDING]

    def register_experiment(self, name: str, model_name: str, resume: bool):
        """Registers a training experiment in Deci's backend.
        :param name:        Name of the experiment to register
        :param model_name:  Name of the model architecture to connect the experiment to
        """
        try:
            self.lab_client.register_user_architecture(name=model_name)
        except (ApiException, ApiTypeError) as e:
            logger.debug(f"The model was already registered, or validation error: {e}")

        self.lab_client.register_experiment(name=name, model_name=model_name, resume=resume)

    def save_experiment_file(self, file_path: str):
        """
        Uploads a training related file to Deci's location in S3. This can be a TensorBoard file or a log
        :params file_path: The local path of the file to be uploaded
        """
        self.lab_client.save_experiment_file(file_path=file_path)

    def upload_file_to_s3(self, tag: str, level: "SentryLevel", from_path: str):
        """Upload a file to the platform S3 bucket.

        :param tag:         Tag that will be associated to the file.
        :param level:       Logging level that will be used to notify the monitoring system that the file was uploaded.
        :param from_path:   Path of the file to upload.
        """
        data = self.lab_client.upload_log_url(tag=tag, level=level)
        signed_url = S3SignedUrl(**data)
        self.lab_client.upload_file_to_s3(from_path=from_path, s3_signed_url=signed_url)


def _move_file_to_folder(src_file_path: str, dest_dir_name: str) -> str:
    """Move a file to a newly created folder in the same directory.

    :param src_file_path:   Path of the file to be moved.
    :param dest_dir_name:   Name of the destination folder.
    :return:                The path of the moved file.
    """
    src_dir_path = os.path.dirname(src_file_path)

    dest_dir_path = os.path.join(src_dir_path, dest_dir_name)
    dest_file_path = os.path.join(dest_dir_path, os.path.basename(src_file_path))

    os.makedirs(dest_dir_path, exist_ok=True)
    shutil.copyfile(src_file_path, dest_file_path)
    return dest_file_path
