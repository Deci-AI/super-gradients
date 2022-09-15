import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

try:
    from pkg_resources import DistributionNotFound
except (ImportError, NameError):
    DistributionNotFound = Exception


class DeciClient:
    def __init__(self):
        from deci_lab_client.client import DeciPlatformClient

        self.lab_client = DeciPlatformClient()
        GlobalHydra.instance().clear()
        self.super_gradients_version = None
        try:
            import pkg_resources

            self.super_gradients_version = pkg_resources.get_distribution("super_gradients").version
        except DistributionNotFound:
            self.super_gradients_version = "3.0.0"

    def _get_file(self, model_name: str, file_name: str) -> str:
        from deci_common.data_interfaces.files_data_interface import FilesDataInterface

        response = self.lab_client.get_autonac_model_file_link(
            model_name=model_name, file_name=file_name, super_gradients_version=self.super_gradients_version
        )
        download_link = response.data
        return FilesDataInterface.download_temporary_file(file_url=download_link)

    def get_model_arch_params(self, model_name: str) -> DictConfig:
        from deci_lab_client.models import AutoNACFileName

        file = self._get_file(model_name=model_name, file_name=AutoNACFileName.STRUCTURE_YAML)
        split_file = file.split("/")
        with hydra.initialize_config_dir(config_dir=f"{'/'.join(split_file[:-1])}/", version_base=None):
            cfg = hydra.compose(config_name=split_file[-1])
        return cfg

    def get_model_weights(self, model_name: str) -> str:
        from deci_lab_client.models import AutoNACFileName

        return self._get_file(model_name=model_name, file_name=AutoNACFileName.WEIGHTS_PTH)
