import os
from typing import Optional

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.sg_loggers.base_sg_logger import BaseSGLogger
from super_gradients.common.environment.ddp_utils import multi_process_safe

logger = get_logger(__name__)

try:
    from deci_lab_client.client import DeciPlatformClient

    _imported_deci_lab_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.debug("Failed to import deci_lab_client")
    _imported_deci_lab_failure = import_err

TENSORBOARD_EVENTS_PREFIX = "events.out.tfevents"
LOGS_PREFIX = "log_"


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
        tb_files_user_prompt: bool = False,
        launch_tensorboard: bool = False,
        tensorboard_port: int = None,
        save_checkpoints_remote: bool = True,
        save_tensorboard_remote: bool = True,
        save_logs_remote: bool = True,
        monitor_system: bool = True,
        model_name: Optional[str] = None,
    ):

        if _imported_deci_lab_failure is not None:
            raise _imported_deci_lab_failure

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

        self.platform_client = DeciPlatformClient()
        self.platform_client.login(token=os.getenv("DECI_PLATFORM_TOKEN"))
        if model_name is None:
            logger.warning(
                "'model_name' parameter not passed. "
                "The experiment won't be connected to an architecture in the Deci platform. "
                "To pass a model_name, please use the 'sg_logger_params.model_name' field in the training recipe."
            )
        self.platform_client.register_experiment(name=experiment_name, model_name=model_name if model_name else None)
        self.checkpoints_dir_path = checkpoints_dir_path

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
        self._upload_latest_file_starting_with(start_with=LOGS_PREFIX)

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
        self.platform_client.save_experiment_file(file_path=most_recent_file_path)
        logger.info(f"File saved to Deci platform: {most_recent_file_path}")
