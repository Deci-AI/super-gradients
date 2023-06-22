import os
from pathlib import Path
from typing import Optional

import torch

from super_gradients.common.registry.registry import register_sg_logger
from super_gradients.common.abstractions.abstract_logger import get_logger

from super_gradients.common.sg_loggers.base_sg_logger import BaseSGLogger
from super_gradients.common.environment.ddp_utils import multi_process_safe
from super_gradients.common.sg_loggers.time_units import TimeUnit

logger = get_logger(__name__)

try:
    import dagshub
    from dagshub.upload import Repo

    _import_dagshub_error = None
except (ModuleNotFoundError, ImportError, NameError) as dagshub_import_err:
    _import_dagshub_error = dagshub_import_err

try:
    import mlflow

    _import_mlflow_error = None
except (ModuleNotFoundError, ImportError, NameError) as mlflow_import_err:
    _import_mlflow_error = mlflow_import_err


@register_sg_logger("dagshub_sg_logger")
class DagsHubSGLogger(BaseSGLogger):
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
        monitor_system: bool = None,
        dagshub_repository: Optional[str] = None,
        log_mlflow_only: bool = False,
    ):
        """

        :param experiment_name:         Name used for logging and loading purposes
        :param storage_location:        If set to 's3' (i.e. s3://my-bucket) saves the Checkpoints in AWS S3 otherwise saves the Checkpoints Locally
        :param resumed:                 If true, then old tensorboard files will **NOT** be deleted when tb_files_user_prompt=True
        :param training_params:         training_params for the experiment.
        :param checkpoints_dir_path:    Local root directory path where all experiment logging directories will reside.
        :param tb_files_user_prompt:    Asks user for Tensorboard deletion prompt.
        :param launch_tensorboard:      Whether to launch a TensorBoard process.
        :param tensorboard_port:        Specific port number for the tensorboard to use when launched (when set to None,
                                        some free port number will be used
        :param save_checkpoints_remote: Saves checkpoints in s3 and DagsHub.
        :param save_tensorboard_remote: Saves tensorboard in s3.
        :param save_logs_remote:        Saves log files in s3 and DagsHub.
        :param monitor_system:          Save the system statistics (GPU utilization, CPU, ...) in the tensorboard
        :param dagshub_repository:      Format: <dagshub_username>/<dagshub_reponame> format is set correctly to avoid
                                        any potential issues. If you are utilizing the dagshub_sg_logger, please specify
                                        the dagshub_repository in sg_logger_params to prevent any interruptions from
                                        prompts during automated pipelines. In the event that the repository does not
                                        exist, it will be created automatically on your behalf.
        :param log_mlflow_only:         Skip logging to DVC, use MLflow for all artifacts being logged
        """
        if monitor_system is not None:
            logger.warning("monitor_system not available on DagsHubSGLogger. To remove this warning, please don't set monitor_system in your logger parameters")

        self.s3_location_available = storage_location.startswith("s3")
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
            save_checkpoints_remote=self.s3_location_available,
            save_tensorboard_remote=self.s3_location_available,
            save_logs_remote=self.s3_location_available,
            monitor_system=False,
        )
        if _import_dagshub_error:
            raise _import_dagshub_error

        if _import_mlflow_error:
            raise _import_mlflow_error

        self.repo_name, self.repo_owner, self.remote = None, None, None
        if dagshub_repository:
            self.repo_name, self.repo_owner = self.splitter(dagshub_repository)

        dagshub_auth = os.getenv("DAGSHUB_USER_TOKEN")
        if dagshub_auth:
            dagshub.auth.add_app_token(dagshub_auth)

        self._init_env_dependency()

        self.log_mlflow_only = log_mlflow_only
        self.save_checkpoints_dagshub = save_checkpoints_remote
        self.save_logs_dagshub = save_logs_remote

    @staticmethod
    def splitter(repo):
        splitted = repo.split("/")
        if len(splitted) != 2:
            raise ValueError(f"Invalid input, should be owner_name/repo_name, but got {repo} instead")
        return splitted[1], splitted[0]

    def _init_env_dependency(self):
        """
        The function creates paths for the DVC directory, models, and artifacts, obtains an authentication token from
        Dagshub, and sets MLflow tracking credentials. It also checks whether the repository name and owner have been
        set and prompts the user to enter them if they haven't. If the remote URI is not set or does not include
        "dagshub", Dagshub is initialized with the repository name and owner, and the remote URI is obtained. The method
        then creates a Repo object with the repository information and sets the DVC folder to the DVC directory path.
        """

        self.paths = {
            "dvc_directory": Path("artifacts"),
            "models": Path("models"),
            "artifacts": Path("artifacts"),
        }

        token = dagshub.auth.get_token()
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        # Check mlflow environment variable is set:
        if not self.repo_name or not self.repo_owner:
            self.repo_name, self.repo_owner = self.splitter(input("Please insert your repository owner_name/repo_name:"))

        if not self.remote or "dagshub" not in os.getenv("MLFLOW_TRACKING_URI"):
            dagshub.init(repo_name=self.repo_name, repo_owner=self.repo_owner)
            self.remote = os.getenv("MLFLOW_TRACKING_URI")

        self.repo = Repo(
            owner=self.remote.split(os.sep)[-2],
            name=self.remote.split(os.sep)[-1].replace(".mlflow", ""),
            branch=os.getenv("BRANCH", "main"),
        )
        self.dvc_folder = self.repo.directory(str(self.paths["dvc_directory"]))

        mlflow.set_tracking_uri(self.remote)
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(nested=True)
        return self.run

    @multi_process_safe
    def _dvc_add(self, local_path="", remote_path=""):
        if not os.path.isfile(local_path):
            FileExistsError(f"Invalid file path: {local_path}")
        self.dvc_folder.add(file=local_path, path=remote_path)

    @multi_process_safe
    def _dvc_commit(self, commit=""):
        self.dvc_folder.commit(commit, versioning="dvc", force=True)

    @multi_process_safe
    def add_config(self, tag: str, config: dict):
        super(DagsHubSGLogger, self).add_config(tag=tag, config=config)
        param_keys = config.keys()
        for pk in param_keys:
            for k, v in config[pk].items():
                try:
                    mlflow.log_params({k: v})
                except Exception:
                    logger.warning(f"Skip to log {k}: {v}")

    @multi_process_safe
    def add_scalar(self, tag: str, scalar_value: float, global_step: [int, TimeUnit] = 0):
        super(DagsHubSGLogger, self).add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)
        if isinstance(global_step, TimeUnit):
            global_step = global_step.get_value()
        mlflow.log_metric(key=tag, value=scalar_value, step=global_step)

    @multi_process_safe
    def add_scalars(self, tag_scalar_dict: dict, global_step: int = 0):
        super(DagsHubSGLogger, self).add_scalars(tag_scalar_dict=tag_scalar_dict, global_step=global_step)
        mlflow.log_metrics(metrics=tag_scalar_dict, step=global_step)

    @multi_process_safe
    def close(self):
        super().close()
        try:
            if not self.log_mlflow_only:
                self._dvc_commit(commit=f"Adding all artifacts from run {mlflow.active_run().info.run_id}")
            mlflow.end_run()
        except Exception:
            pass

    @multi_process_safe
    def add_file(self, file_name: str = None):
        super().add_file(file_name)
        if self.log_mlflow_only:
            mlflow.log_artifact(file_name)
        else:
            self._dvc_add(local_path=file_name, remote_path=os.path.join(self.paths["artifacts"], self.experiment_log_path))

    @multi_process_safe
    def upload(self):
        super().upload()

        if self.save_logs_dagshub:
            if self.log_mlflow_only:
                mlflow.log_artifact(self.experiment_log_path)
            else:
                self._dvc_add(local_path=self.experiment_log_path, remote_path=os.path.join(self.paths["artifacts"], self.experiment_log_path))

    @multi_process_safe
    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = 0):
        name = f"ckpt_{global_step}.pth" if tag is None else tag
        if not name.endswith(".pth"):
            name += ".pth"
        path = os.path.join(self._local_dir, name)
        torch.save(state_dict, path)
        if self.save_checkpoints_dagshub:
            mlflow.log_artifact(path)
            if (global_step >= (self.max_global_steps - 1)) and not self.log_mlflow_only:
                self._dvc_add(local_path=path, remote_path=os.path.join(self.paths["models"], name))
