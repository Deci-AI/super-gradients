import os

from typing import Union, Any

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch


from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry.registry import register_sg_logger
from super_gradients.common.sg_loggers.base_sg_logger import BaseSGLogger
from super_gradients.common.environment.ddp_utils import multi_process_safe
from super_gradients.common.sg_loggers.time_units import TimeUnit

logger = get_logger(__name__)

try:
    from clearml import Task

    _imported_clear_ml_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.debug("Failed to import clearml")
    _imported_clear_ml_failure = import_err


@register_sg_logger("clearml_sg_logger")
class ClearMLSGLogger(BaseSGLogger):
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
    ):
        """
        :param project_name:            ClearML project name that can include many experiments
        :param experiment_name:         Name used for logging and loading purposes
        :param storage_location:        If set to 's3' (i.e. s3://my-bucket) saves the Checkpoints in AWS S3 otherwise saves the Checkpoints Locally
        :param resumed:                 If true, then old tensorboard files will **NOT** be deleted when tb_files_user_prompt=True
        :param training_params:         training_params for the experiment.
        :param checkpoints_dir_path:    Local root directory path where all experiment logging directories will reside.
        :param tb_files_user_prompt:    Asks user for Tensorboard deletion prompt.
        :param launch_tensorboard:      Whether to launch a TensorBoard process.
        :param tensorboard_port:        Specific port number for the tensorboard to use when launched (when set to None, some free port number will be used
        :param save_checkpoints_remote: Saves checkpoints in s3.
        :param save_tensorboard_remote: Saves tensorboard in s3.
        :param save_logs_remote:        Saves log files in s3.
        :param monitor_system:          Not Available for ClearML logger. Save the system statistics (GPU utilization, CPU, ...) in the tensorboard
        """
        if monitor_system is not None:
            logger.warning("monitor_system not available on ClearMLSGLogger. To remove this warning, please don't set monitor_system in your logger parameters")

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

        if _imported_clear_ml_failure is not None:
            raise _imported_clear_ml_failure

        self.setup(project_name, experiment_name)

        self.save_checkpoints = save_checkpoints_remote
        self.save_tensorboard = save_tensorboard_remote
        self.save_logs = save_logs_remote

    @multi_process_safe
    def setup(self, project_name, experiment_name):
        from multiprocessing.process import BaseProcess

        # Prevent clearml modifying os.fork and BaseProcess.run, which can cause a DataLoader to crash (if num_worker > 0)
        # Issue opened here: https://github.com/allegroai/clearml/issues/790
        default_fork, default_run = os.fork, BaseProcess.run
        self.task = Task.init(
            project_name=project_name,  # project name of at least 3 characters
            task_name=experiment_name,  # task name of at least 3 characters
            continue_last_task=0,  # This prevents clear_ml to add an offset to the epoch
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False,
            auto_resource_monitoring=False,
            auto_connect_streams=True,
        )
        os.fork, BaseProcess.run = default_fork, default_run
        self.clearml_logger = self.task.get_logger()

    @multi_process_safe
    def add_config(self, tag: str, config: dict):
        super(ClearMLSGLogger, self).add_config(tag=tag, config=config)
        self.task.connect(config)

    def __add_scalar(self, tag: str, scalar_value: float, global_step: int):
        self.clearml_logger.report_scalar(title=tag, series=tag, value=scalar_value, iteration=global_step)

    @multi_process_safe
    def add_scalar(self, tag: str, scalar_value: float, global_step: Union[int, TimeUnit] = 0):
        super(ClearMLSGLogger, self).add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)
        if isinstance(global_step, TimeUnit):
            global_step = global_step.get_value()
        self.__add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)

    @multi_process_safe
    def add_scalars(self, tag_scalar_dict: dict, global_step: int = 0):
        super(ClearMLSGLogger, self).add_scalars(tag_scalar_dict=tag_scalar_dict, global_step=global_step)
        for tag, scalar_value in tag_scalar_dict.items():
            self.__add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)

    def __add_image(
        self,
        tag: str,
        image: Union[torch.Tensor, np.array, Image.Image],
        global_step: int,
    ):
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        if image.shape[0] < 5:
            image = image.transpose([1, 2, 0])
        self.clearml_logger.report_image(
            title=tag,
            series=tag,
            image=image,
            iteration=global_step,
            max_image_history=-1,
        )

    @multi_process_safe
    def add_image(
        self,
        tag: str,
        image: Union[torch.Tensor, np.array, Image.Image],
        data_format="CHW",
        global_step: int = 0,
    ):
        super(ClearMLSGLogger, self).add_image(tag=tag, image=image, data_format=data_format, global_step=global_step)
        self.__add_image(tag, image, global_step)

    @multi_process_safe
    def add_images(
        self,
        tag: str,
        images: Union[torch.Tensor, np.array],
        data_format="NCHW",
        global_step: int = 0,
    ):
        super(ClearMLSGLogger, self).add_images(tag=tag, images=images, data_format=data_format, global_step=global_step)
        for image in images:
            self.__add_image(tag, image, global_step)

    @multi_process_safe
    def add_video(self, tag: str, video: Union[torch.Tensor, np.array], global_step: int = 0):
        super().add_video(tag, video, global_step)
        logger.warning("ClearMLSGLogger does not support uploading video to clearML from a tensor/array.")

    @multi_process_safe
    def add_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, np.array],
        bins: str,
        global_step: int = 0,
    ):
        super().add_histogram(tag, values, bins, global_step)
        self.clearml_logger.report_histogram(title=tag, series=tag, iteration=global_step, values=values)

    @multi_process_safe
    def add_text(self, tag: str, text_string: str, global_step: int = 0):
        super().add_text(tag, text_string, global_step)
        self.clearml_logger.report_text(text_string)

    @multi_process_safe
    def add_figure(self, tag: str, figure: plt.figure, global_step: int = 0):
        super().add_figure(tag, figure, global_step)
        name = f"tmp_{tag}.png"
        path = os.path.join(self._local_dir, name)
        figure.savefig(path)
        self.task.upload_artifact(name=name, artifact_object=path)
        os.remove(path)

    @multi_process_safe
    def close(self):
        super().close()
        self.task.close()

    @multi_process_safe
    def add_file(self, file_name: str = None):
        super().add_file(file_name)
        self.task.upload_artifact(name=file_name, artifact_object=os.path.join(self._local_dir, file_name))

    @multi_process_safe
    def upload(self):
        super().upload()

        if self.save_tensorboard:
            name = self._get_tensorboard_file_name().split("/")[-1]
            self.task.upload_artifact(name=name, artifact_object=self._get_tensorboard_file_name())

        if self.save_logs:
            name = self.experiment_log_path.split("/")[-1]
            self.task.upload_artifact(name=name, artifact_object=self.experiment_log_path)

    @multi_process_safe
    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = 0):
        name = f"ckpt_{global_step}.pth" if tag is None else tag
        if not name.endswith(".pth"):
            name += ".pth"

        path = os.path.join(self._local_dir, name)
        torch.save(state_dict, path)

        if self.save_checkpoints:
            if self.s3_location_available:
                self.model_checkpoints_data_interface.save_remote_checkpoints_file(self.experiment_name, self._local_dir, name)
            self.task.upload_artifact(name=name, artifact_object=path)

    def _get_tensorboard_file_name(self):
        try:
            tb_file_path = self.tensorboard_writer.file_writer.event_writer._file_name
        except RuntimeError:
            logger.warning("tensorboard file could not be located for ")
            return None

        return tb_file_path

    def add(self, tag: str, obj: Any, global_step: int = None):
        pass
