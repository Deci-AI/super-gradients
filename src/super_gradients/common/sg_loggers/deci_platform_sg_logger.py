import os
import io
from contextlib import contextmanager

from super_gradients.common.registry.registry import register_sg_logger
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.sg_loggers.base_sg_logger import BaseSGLogger, EXPERIMENT_LOGS_PREFIX, LOGGER_LOGS_PREFIX, CONSOLE_LOGS_PREFIX
from super_gradients.common.environment.ddp_utils import multi_process_safe
from super_gradients.common.plugins.deci_client import DeciClient
from contextlib import redirect_stdout

logger = get_logger(__name__)


TENSORBOARD_EVENTS_PREFIX = "events.out.tfevents"


@register_sg_logger("deci_platform_sg_logger")
class DeciPlatformSGLogger(BaseSGLogger):
    """Logger responsible to push logs and tensorboard artifacts to Deci platform."""

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        storage_location: str,
        resumed: bool,
        training_params: dict,
        checkpoints_dir_path: str,
        model_name: str,
        upload_model: bool = True,
        tb_files_user_prompt: bool = False,
        launch_tensorboard: bool = False,
        tensorboard_port: int = None,
        save_checkpoints_remote: bool = True,
        save_tensorboard_remote: bool = True,
        save_logs_remote: bool = True,
        monitor_system: bool = True,
    ):
        """

        :param experiment_name:         Name used for logging and loading purposes
        :param storage_location:        If set to 's3' (i.e. s3://my-bucket) saves the Checkpoints in AWS S3 otherwise saves the Checkpoints Locally
        :param resumed:                 If true, then old tensorboard files will **NOT** be deleted when tb_files_user_prompt=True
        :param training_params:         training_params for the experiment.
        :param checkpoints_dir_path:    Local root directory path where all experiment logging directories will reside.
        :param model_name:              Name of the model to be used for logging.
        :param upload_model:            Whether to upload the model to the Deci Platform or not.
        :param tb_files_user_prompt:    Asks user for Tensorboard deletion prompt.
        :param launch_tensorboard:      Whether to launch a TensorBoard process.
        :param tensorboard_port:        Specific port number for the tensorboard to use when launched (when set to None, some free port number will be used
        :param save_checkpoints_remote: Saves checkpoints in s3.
        :param save_tensorboard_remote: Saves tensorboard in s3.
        :param save_logs_remote:        Saves log files in s3.
        :param monitor_system:          Save the system statistics (GPU utilization, CPU, ...) in the tensorboard
        """
        super().__init__(
            project_name=project_name,
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
            save_logs_remote=save_logs_remote,
            monitor_system=monitor_system,
        )
        self.platform_client = DeciClient()
        self.platform_client.register_experiment(name=experiment_name, model_name=model_name if model_name else None, resume=resumed)
        self.checkpoints_dir_path = checkpoints_dir_path
        self.upload_model = upload_model

    @multi_process_safe
    def upload(self):
        """
        Upload both to the destination specified by the user (base behavior), and to Deci platform.
        """
        # Upload to the destination specified by the user
        super(DeciPlatformSGLogger, self).upload()

        # Upload to Deci platform
        if not os.path.isdir(self.checkpoints_dir_path):
            raise ValueError("Provided directory does not exist")

        self._upload_latest_file_starting_with(start_with=TENSORBOARD_EVENTS_PREFIX)
        self._upload_latest_file_starting_with(start_with=EXPERIMENT_LOGS_PREFIX)
        self._upload_latest_file_starting_with(start_with=LOGGER_LOGS_PREFIX)
        self._upload_latest_file_starting_with(start_with=CONSOLE_LOGS_PREFIX)
        self._upload_folder_files(folder_name=".hydra")

    @multi_process_safe
    def _save_checkpoint(self, path: str, state_dict: dict) -> None:
        """Save the Checkpoint locally, and then upload it to Deci platform if required.

        :param path:        Full path of the checkpoint
        :param state_dict:  State dict of the checkpoint
        """
        super(DeciPlatformSGLogger, self)._save_checkpoint(path=path, state_dict=state_dict)
        if self.upload_model:
            self._save_experiment_file(file_path=path)

    @multi_process_safe
    def _upload_latest_file_starting_with(self, start_with: str):
        """
        Upload the most recent file starting with a specific prefix to the Deci platform.

        :param start_with: prefix of the file to upload
        """

        files_path = [
            os.path.join(self.checkpoints_dir_path, file_name) for file_name in os.listdir(self.checkpoints_dir_path) if file_name.startswith(start_with)
        ]

        most_recent_file_path = max(files_path, key=os.path.getctime)
        self._save_experiment_file(file_path=most_recent_file_path)

    @multi_process_safe
    def _upload_folder_files(self, folder_name: str):
        """
        Upload all the files of a given folder.

        :param folder_name: Name of the folder that contains the files to upload
        """
        folder_path = os.path.join(self.checkpoints_dir_path, folder_name)

        if not os.path.exists(folder_path):
            return

        for file in os.listdir(folder_path):
            self._save_experiment_file(file_path=f"{folder_path}/{file}")

    def _save_experiment_file(self, file_path: str):
        with log_stdout():  # TODO: remove when platform_client remove prints from save_experiment_file
            self.platform_client.save_experiment_file(file_path=file_path)
        logger.info(f"File saved to Deci platform: {file_path}")


@contextmanager
def log_stdout():
    """Redirect stdout to DEBUG."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        yield

    redirected_str = buffer.getvalue()
    if redirected_str:
        logger.debug(msg=redirected_str)
