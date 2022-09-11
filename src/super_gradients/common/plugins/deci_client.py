import boto3
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

class MockClient:

    def __init__(self):
        self.bucket_name = 'deci-model-repository-research'
        self.prefix = 'mock_client'
        self.s3 = boto3.client('s3')
        GlobalHydra.instance().clear()

    def get_model_arch_params(self, model_name: str) -> DictConfig:
        tmp_file_path = '/home/ofri/arch_params.yaml'
        self.s3.download_file(self.bucket_name, f'{self.prefix}/{model_name}/arch_params.yaml', tmp_file_path)
        with hydra.initialize_config_dir(config_dir='/home/ofri/'):
            cfg = hydra.compose(config_name='arch_params.yaml')
        return cfg

    def get_model_weights(self, model_name: str) -> DictConfig:
        tmp_file_path = '/home/ofri/weights.pth'
        self.s3.download_file(self.bucket_name, f'{self.prefix}/{model_name}/{model_name}.pth', tmp_file_path)
        return tmp_file_path
