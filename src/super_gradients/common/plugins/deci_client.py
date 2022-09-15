import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from super_gradients import get_version
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)

client_enabled = True
try:
    from deci_lab_client.client import DeciPlatformClient
    from deci_common.data_interfaces.files_data_interface import FilesDataInterface
    from deci_lab_client.models import AutoNACFileName
except (ImportError, NameError):
    client_enabled = False

DECI_LAB_CLIENT_DNS = "api.development.deci.ai"


class DeciClient:
    """
    A client to deci platform and model zoo.
    requires credentials for connection
    """

    def __init__(self):
        if not client_enabled:
            logger.warning('deci-lab-client or deci-common are not installed. Model cannot be loaded from deci lab.'
                           'Please install deci-lab-client>=2.55.0 and deci-common>=3.4.1')

        self.lab_client = DeciPlatformClient(api_host=DECI_LAB_CLIENT_DNS)
        GlobalHydra.instance().clear()
        self.super_gradients_version = None
        self.super_gradients_version = get_version()

    def _get_file(self, model_name: str, file_name: str) -> str:
        response = self.lab_client.get_autonac_model_file_link(
            model_name=model_name, file_name=file_name, super_gradients_version=self.super_gradients_version
        )
        download_link = response.data
        return FilesDataInterface.download_temporary_file(file_url=download_link)

    def get_model_arch_params(self, model_name: str) -> DictConfig:
        file = self._get_file(model_name=model_name, file_name=AutoNACFileName.STRUCTURE_YAML)
        split_file = file.split("/")
        with hydra.initialize_config_dir(config_dir=f"{'/'.join(split_file[:-1])}/", version_base=None):
            cfg = hydra.compose(config_name=split_file[-1])
        return cfg

    def get_model_weights(self, model_name: str) -> str:
        return self._get_file(model_name=model_name, file_name=AutoNACFileName.WEIGHTS_PTH)
