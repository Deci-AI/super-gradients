import hydra
from deci_common.data_interfaces.files_data_interface import FilesDataInterface
from deci_lab_client import AutoNACFileName
from deci_lab_client.client import DeciPlatformClient
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


class DeciClient:
    def __init__(self, lab_client: DeciPlatformClient):
        self.lab_client = lab_client
        GlobalHydra.instance().clear()
        self.super_gradients_version = None
        try:
            import pkg_resources

            self.super_gradients_version = pkg_resources.get_distribution("super_gradients").version
        except Exception as e:
            print(e)

    def _get_file(self, model_name: str, file_name: str) -> str:
        response = self.lab_client.get_autonac_model_file_link(model_name=model_name, file_name=file_name)
        download_link = response.data
        return FilesDataInterface.download_temporary_file(file_url=download_link)

    def get_model_arch_params(self, model_name: str) -> DictConfig:
        file = self._get_file(model_name=model_name, file_name=AutoNACFileName.STRUCTURE_YAML)
        split_file = file.split("/")
        with hydra.initialize_config_dir(config_dir=f"/{'/'.join(split_file)}/"):
            cfg = hydra.compose(config_name=split_file[-1])
        return cfg

    def get_model_weights(self, model_name: str) -> str:
        return self._get_file(model_name=model_name, file_name=AutoNACFileName.WEIGHTS_PTH)
