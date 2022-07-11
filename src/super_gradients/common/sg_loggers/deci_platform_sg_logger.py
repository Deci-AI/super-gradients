import os

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.sg_loggers.base_sg_logger import BaseSGLogger
from super_gradients.common.environment.env_helpers import multi_process_safe
from super_gradients.training.params import TrainingParams

logger = get_logger(__name__)

try:
    from deci_lab_client.client import DeciPlatformClient
    _imported_deci_lab_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.warn("Failed to import deci_lab_client")
    _imported_deci_lab_failure = import_err

TENSORBOARD_EVENTS_PREFIX = 'events.out.tfevents'
LOGS_PREFIX = 'log_'


class DeciPlatformSGLogger(BaseSGLogger):
    """Logger responsible to push logs and tensorboard artifacts to Deci platform."""

    def __init__(self, **kwargs):

        if _imported_deci_lab_failure is not None:
            raise _imported_deci_lab_failure

        auth_token = os.getenv("DECI_PLATFORM_TOKEN")
        if auth_token is None:
            raise ValueError('The environment variable "DECI_PLATFORM_TOKEN" is required in order to use '
                             'DeciPlatformSGLogger. Please set it with your own credentials '
                             '(available in https://console.deci.ai/settings)')

        super().__init__(**kwargs)
        self.platform_client = DeciPlatformClient()
        self.platform_client.login(token=auth_token)
        self.platform_client.register_experiment(name=kwargs["experiment_name"])
        self.checkpoints_dir_path = kwargs["checkpoints_dir_path"]

    @multi_process_safe
    def upload(self):
        """
        Upload both to the destination specified by the user (base behavior), and to Deci platform.
        """
        # Upload to the destination specified by the user
        super(DeciPlatformSGLogger, self).upload()

        # Upload to Deci platform
        if not os.path.isdir(self.checkpoints_dir_path):
            raise ValueError('Provided directory does not exist')

        for file_name in os.listdir(self.checkpoints_dir_path):
            if file_name.startswith(TENSORBOARD_EVENTS_PREFIX) or file_name.startswith(LOGS_PREFIX):
                upload_success = self.platform_client.save_experiment_file(
                    file_path=f"{self.checkpoints_dir_path}/{file_name}")

                if not upload_success:
                    logger.error(f'Failed to upload to platform : "{file_name}" ')
