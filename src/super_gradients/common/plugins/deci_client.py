import json

import hydra
import pkg_resources
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)

client_enabled = True
try:
    from deci_lab_client.client import DeciPlatformClient
    from deci_common.data_interfaces.files_data_interface import FilesDataInterface
    from deci_lab_client.models import AutoNACFileName
    from deci_lab_client import ApiException
except (ImportError, NameError):
    client_enabled = False


class DeciClient:
    """
    A client to deci platform and model zoo.
    requires credentials for connection
    """

    def __init__(self):
        if not client_enabled:
            logger.error('deci-lab-client or deci-common are not installed. Model cannot be loaded from deci lab.'
                         'Please install deci-lab-client>=2.55.0 and deci-common>=3.4.1')
            return

        self.lab_client = DeciPlatformClient('api.staging.deci.ai')
        GlobalHydra.instance().clear()
        self.super_gradients_version = None
        try:
            self.super_gradients_version = pkg_resources.get_distribution("super_gradients").version
        except pkg_resources.DistributionNotFound:
            self.super_gradients_version = "3.0.0"

    def _get_file(self, model_name: str, file_name: str) -> str:
        try:
            response = self.lab_client.get_autonac_model_file_link(
                model_name=model_name, file_name=file_name, super_gradients_version=self.super_gradients_version
            )
            download_link = response.data
        except ApiException as e:
            if e.status == 401:
                logger.error("Deci client: Unauthorized. wrong token or token was not defined. please login to deci-lab-client "
                             "by calling DeciPlatformClient().login(<token>)")
            elif e.status == 400 and e.body is not None and "message" in e.body:
                logger.error(f"Deci client: {json.loads(e.body)['message']}")
            else:
                logger.error(e.body)

            return None
        return FilesDataInterface.download_temporary_file(file_url=download_link)

    def get_model_arch_params(self, model_name: str) -> DictConfig:
        if not client_enabled:
            return None

        file = self._get_file(model_name=model_name, file_name=AutoNACFileName.STRUCTURE_YAML)
        if file is None:
            return None

        split_file = file.split("/")
        with hydra.initialize_config_dir(config_dir=f"{'/'.join(split_file[:-1])}/", version_base=None):
            cfg = hydra.compose(config_name=split_file[-1])
        return cfg

    def get_model_weights(self, model_name: str) -> str:
        if not client_enabled:
            return None

        return self._get_file(model_name=model_name, file_name=AutoNACFileName.WEIGHTS_PTH)
