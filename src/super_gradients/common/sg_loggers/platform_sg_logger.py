import os
import sys

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


class PlatformSGLogger(BaseSGLogger):

    def __init__(self,
                 project_name: str,
                 experiment_name: str,
                 storage_location: str,
                 resumed: bool,
                 training_params: TrainingParams,
                 checkpoints_dir_path: str,
                 auth_token: str,
                 tb_files_user_prompt: bool = False,
                 launch_tensorboard: bool = False,
                 tensorboard_port: int = None,
                 save_checkpoints_remote: bool = True,
                 save_tensorboard_remote: bool = True,
                 save_logs_remote: bool = True):
        """
        Logger responsible to push tensorboard to Deci platform.

        :param experiment_name:         Used for logging and loading purposes
        :param storage_location:        If set to 's3' (i.e. s3://my-bucket) saves the Checkpoints in AWS S3
                                            otherwise saves the Checkpoints Locally
        :param resumed:                 if true, then old tensorboard files will *not* be deleted when
                                            tb_files_user_prompt=True
        :param training_params:         training_params for the experiment.
        :param checkpoints_dir_path:    Local root directory path where all experiment logging directories will reside.
        :param auth_token:              Deci platform authorization token (avalaible on https://console.deci.ai/)
        :param tb_files_user_prompt:    Asks user for Tensorboard deletion prompt.
        :param launch_tensorboard:      Whether to launch a TensorBoard process.
        :param tensorboard_port:        Specific port number for the tensorboard to use when launched
                                            (when set to None, some free port number will be used)
        :param save_checkpoints_remote: Saves checkpoints in s3.
        :param save_tensorboard_remote: Saves tensorboard in s3.
        :param save_logs_remote:        Saves log files in s3.
        """
        if _imported_deci_lab_failure is not None:
            raise _imported_deci_lab_failure
        super().__init__(project_name=project_name,
                         experiment_name=experiment_name,
                         storage_location=storage_location,
                         resumed=resumed,
                         training_params=training_params,
                         checkpoints_dir_path=checkpoints_dir_path,
                         tb_files_user_prompt=tb_files_user_prompt,
                         launch_tensorboard=launch_tensorboard,
                         tensorboard_port=tensorboard_port,
                         save_checkpoints_remote=save_checkpoints_remote,
                         save_tensorboard_remote=save_tensorboard_remote,
                         save_logs_remote=save_logs_remote)
        self.platform_client = DeciPlatformClient("api.deci.ai", 443, https=True)
        self.platform_client.login(token=auth_token)
        self.checkpoints_dir_path = checkpoints_dir_path
        # self.platform_client.create_experiment()

    @multi_process_safe
    def upload(self):
        """
        Upload both to the destination specified by the user (base behavior), and to Deci platform.
        """
        # Upload to the destination specified by the user
        super(PlatformSGLogger, self).upload()

        # Upload to Deci platform
        if not os.path.isdir(self.checkpoints_dir_path):
            raise ValueError('[' + sys._getframe().f_code.co_name + '] - Provided directory does not exist')

        for tb_events_file_name in os.listdir(self.checkpoints_dir_path):
            if tb_events_file_name.startswith(TENSORBOARD_EVENTS_PREFIX):
                upload_success = FFFF(model_checkpoint_local_dir=self.checkpoints_dir_path,
                                       checkpoints_file_name=tb_events_file_name)

                if not upload_success:
                    logger.error('Failed to upload tb_events_file: ' + tb_events_file_name)
