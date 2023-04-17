import os
import json
import sys
import shutil
from zipfile import ZipFile
from typing import List, Optional, Any

import importlib.util
from omegaconf import DictConfig
from torch import nn

import super_gradients
from super_gradients.common.environment.env_variables import env_variables
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.cfg_utils import load_arch_params, load_recipe

logger = get_logger(__name__)

client_enabled = True
try:
    from deci_lab_client.client import DeciPlatformClient
    from deci_lab_client.types import S3SignedUrl
    from deci_lab_client.models import ModelBenchmarkState
    from deci_common.data_interfaces.files_data_interface import FilesDataInterface
    from deci_lab_client.models import AutoNACFileName
    from deci_lab_client import ApiException, BodyRegisterUserArchitecture

except (ImportError, NameError):
    client_enabled = False


class DeciClient:
    """
    A client to deci platform and model zoo.
    requires credentials for connection
    """

    def __init__(self):
        if not client_enabled:
            logger.error(
                "deci-lab-client or deci-common are not installed. Model cannot be loaded from deci lab."
                "Please install deci-lab-client>=2.55.0 and deci-common>=3.4.1"
            )
            return

        self.api_host = env_variables.DECI_API_HOST
        self.lab_client = DeciPlatformClient(api_host=self.api_host)
        self.lab_client.login(token=env_variables.DECI_PLATFORM_TOKEN)

    def _get_file(self, model_name: str, file_name: str) -> Optional[str]:
        """Get a file from the DeciPlatform if it exists, otherwise returns None
        :param model_name:      Name of the model to download from, as saved in the platform.
        :param file_name:       Name of the file to download
        :return:            Path were the downloaded file was saved to. None if not found.
        """
        try:
            response = self.lab_client.get_autonac_model_file_link(
                model_name=model_name, file_name=file_name, super_gradients_version=super_gradients.__version__
            )
            download_link = response.data
        except ApiException as e:
            if e.status == 401:
                logger.error(
                    "Unauthorized. wrong token or token was not defined. please login to deci-lab-client " "by calling DeciPlatformClient().login(<token>)"
                )
            elif e.status == 400 and e.body is not None and "message" in e.body:
                logger.error(f"Deci client: {json.loads(e.body)['message']}")
            else:
                logger.debug(e.body)
            return None

        file_path = FilesDataInterface.download_temporary_file(file_url=download_link)

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

    def upload_model(self, model: nn.Module, model_meta_data, optimization_request_form):
        """
        This function will upload the trained model to the Deci Lab

        :param model:                     The resulting model from the training process
        :param model_meta_data:           Metadata to accompany the model
        :param optimization_request_form: The optimization parameters
        """
        self.lab_client.add_model(
            add_model_request=model_meta_data,
            optimization_request=optimization_request_form,
            local_loaded_model=model,
        )

    def is_model_benchmarking(self, name: str) -> bool:
        """Check if a given model is still benchmarking or not.
        :param name: The mode name.
        """
        benchmark_state = self.lab_client.get_model_by_name(name=name).data.benchmark_state
        return benchmark_state in [ModelBenchmarkState.IN_PROGRESS, ModelBenchmarkState.PENDING]

    def register_experiment(self, name: str, model_name: str, resume: bool):
        """Registers a training experiment in Deci's backend.
        :param name:        Name of the experiment to register
        :param model_name:  Name of the model architecture to connect the experiment to
        """
        try:
            self.lab_client.register_user_architecture(BodyRegisterUserArchitecture(architecture_name=model_name))
        except ApiException as e:
            if e.status == 422:
                logger.debug(f"The model was already registered, or validation error: {e.body}")
            else:
                raise e

        self.lab_client.register_experiment(name=name, model_name=model_name, resume=resume)

    def save_experiment_file(self, file_path: str):
        """
        Uploads a training related file to Deci's location in S3. This can be a TensorBoard file or a log
        :params file_path: The local path of the file to be uploaded
        """
        self.lab_client.save_experiment_file(file_path=file_path)

    def upload_file_to_s3(self, tag: str, level: str, from_path: str):
        """Upload a file to the platform S3 bucket.

        :param tag:         Tag that will be associated to the file.
        :param level:       Logging level that will be used to notify the monitoring system that the file was uploaded.
        :param from_path:   Path of the file to upload.
        """
        data = self.lab_client.upload_log_url(tag=tag, level=level)
        signed_url = S3SignedUrl(**data.data)
        self.lab_client.upload_file_to_s3(from_path=from_path, s3_signed_url=signed_url)

    def add_model(
        self,
        model_metadata,
        hardware_types: List[str],
        model_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        **kwargs: Any,
    ):
        """Adds a new model to the company's model repository.
        :param model_metadata: The model metadata.
        :param hardware_types: The hardware types you want to benchmark the model on.
        :param model_path:      The path of the model on the local operating system.
        :param model:           Pytorch loaded model object.
                                If your model's framework is pytorch you may pass the following parameters as kwargs in order to control the conversion to onnx
        :param kwargs: Extra arguments to be passed to the PyTorch to ONNX conversion, for example:
            opset_version
            do_constant_folding
            dynamic_axes
            input_names
            output_names
        """

        self.lab_client.add_model_v2(model_metadata=model_metadata, hardware_types=hardware_types, model_path=model_path, model=model, **kwargs)


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
