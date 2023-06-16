import os
from typing import Union, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.ddp_utils import multi_process_safe
from super_gradients.common.environment.env_variables import env_variables
from super_gradients.common.registry.registry import register_sg_logger
from super_gradients.common.sg_loggers.base_sg_logger import BaseSGLogger
from super_gradients.common.sg_loggers.time_units import TimeUnit

logger = get_logger(__name__)

try:
    import wandb
except (ModuleNotFoundError, ImportError, NameError):
    pass  # no action or logging - this is normal in most cases


WANDB_ID_PREFIX = "wandb_id."
WANDB_INCLUDE_FILE_NAME = ".wandbinclude"


@register_sg_logger("wandb_sg_logger")
class WandBSGLogger(BaseSGLogger):
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
        entity: Optional[str] = None,
        api_server: Optional[str] = None,
        save_code: bool = False,
        monitor_system: bool = None,
        save_checkpoint_as_artifact: bool = False,
        **kwargs,
    ):
        """

        :param experiment_name:         Name used for logging and loading purposes
        :param storage_location:        If set to 's3' (i.e. s3://my-bucket) saves the Checkpoints in AWS S3 otherwise saves the Checkpoints Locally
        :param resumed:                 If true, then old tensorboard files will **NOT** be deleted when tb_files_user_prompt=True
        :param training_params:         training_params for the experiment.
        :param checkpoints_dir_path:    Local root directory path where all experiment logging directories will reside.
        :param tb_files_user_prompt:    Asks user for Tensorboard deletion prompt.
        :param launch_tensorboard:      Whether to launch a TensorBoard process.
        :param tensorboard_port:        Specific port number for the tensorboard to use when launched (when set to None, some free port number will be used)
        :param save_checkpoints_remote: Saves checkpoints in s3.
        :param save_tensorboard_remote: Saves tensorboard in s3.
        :param save_logs_remote:        Saves log files in s3.
        :param monitor_system:          Not Available for WandB logger. Save the system statistics (GPU utilization, CPU, ...) in the tensorboard
        :param save_code:               Save current code to wandb
        :save_checkpoint_as_artifact:   Save model checkpoint using Weights & Biases Artifact. Note that setting this option to True would save model
                                        checkpoints every epoch as a versioned artifact, which will result in use of increased storage usage on
                                        Weights & Biases.
        """
        if monitor_system is not None:
            logger.warning("monitor_system not available on WandBSGLogger. To remove this warning, please don't set monitor_system in your logger parameters")

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

        if api_server is not None:
            if api_server != env_variables.WANDB_BASE_URL:
                logger.warning(f"WANDB_BASE_URL environment parameter not set to {api_server}. Setting the parameter")
                os.environ["WANDB_BASE_URL"] = api_server

        # allow passing an arbitrary pre-defined wandb_id
        wandb_id = kwargs.pop("wandb_id", None)

        self.resumed = resumed
        if self.resumed:
            if wandb_id is None:
                if self._resume_from_remote_sg_logger:
                    raise RuntimeError(
                        "For WandB loggers, when training_params.resume_from_remote_sg_logger=True "
                        "pass the run id through the wandb_id arg in sg_logger_params"
                    )
                wandb_id = self._get_wandb_id()

        if wandb.run is None:
            run = wandb.init(project=project_name, name=experiment_name, entity=entity, resume=resumed, id=wandb_id, **kwargs)
        else:
            logger.warning(
                "A Weights & Biases run was initialized before initializing `WandBSGLogger`. "
                "This means that `super-gradients` cannot control the run ID to which this session will be logged."
            )
            logger.warning(f"In order to resume this run please call `wandb.init(id={wandb.run.id}, resume='must')` before reinitializing `WandBSGLogger`.")
            run = wandb.run

        if save_code:
            self._save_code_lines()

        self._set_wandb_id(run.id)
        self.save_checkpoints_wandb = save_checkpoints_remote
        self.save_tensorboard_wandb = save_tensorboard_remote
        self.save_logs_wandb = save_logs_remote
        self.save_checkpoint_as_artifact = save_checkpoint_as_artifact

    @multi_process_safe
    def _save_code_lines(self):
        """
        Save the current code to wandb.
        If a file named .wandbinclude is avilable in the root dir of the project the settings will be taken from the file.
        Otherwise, all python file in the current working dir (recursively) will be saved.
        File structure: a single relative path or a single type in each line.
        i.e:

        src
        tests
        examples
        *.py
        *.yaml

        The paths and types in the file are the paths and types to be included in code upload to wandb
        """
        base_path, paths, types = self._get_include_paths()

        if len(types) > 0:

            def func(path):
                for p in paths:
                    if path.startswith(p):
                        for t in types:
                            if path.endswith(t):
                                return True
                return False

            include_fn = func
        else:
            include_fn = lambda path: path.endswith(".py")

        if base_path != ".":
            wandb.run.log_code(base_path, include_fn=include_fn)
        else:
            wandb.run.log_code(".", include_fn=include_fn)

    @multi_process_safe
    def add_config(self, tag: str, config: dict):
        super(WandBSGLogger, self).add_config(tag=tag, config=config)
        wandb.config.update(config, allow_val_change=self.resumed)

    @multi_process_safe
    def add_scalar(self, tag: str, scalar_value: float, global_step: Union[int, TimeUnit] = 0):
        super(WandBSGLogger, self).add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)
        if isinstance(global_step, TimeUnit):
            wandb.log(data={tag: scalar_value, global_step.get_name(): global_step.get_value()})
        else:
            wandb.log(data={tag: scalar_value}, step=global_step)

    @multi_process_safe
    def add_scalars(self, tag_scalar_dict: dict, global_step: int = 0):
        super(WandBSGLogger, self).add_scalars(tag_scalar_dict=tag_scalar_dict, global_step=global_step)
        wandb.log(data=tag_scalar_dict, step=global_step)

    @multi_process_safe
    def add_image(self, tag: str, image: Union[torch.Tensor, np.array, Image.Image], data_format="CHW", global_step: int = 0):
        super(WandBSGLogger, self).add_image(tag=tag, image=image, data_format=data_format, global_step=global_step)
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        if image.shape[0] < 5:
            image = image.transpose([1, 2, 0])
        wandb.log(data={tag: wandb.Image(image, caption=tag)}, step=global_step)

    @multi_process_safe
    def add_images(self, tag: str, images: Union[torch.Tensor, np.array], data_format="NCHW", global_step: int = 0):
        super(WandBSGLogger, self).add_images(tag=tag, images=images, data_format=data_format, global_step=global_step)

        wandb_images = []
        for im in images:
            if isinstance(im, torch.Tensor):
                im = im.cpu().detach().numpy()

            if im.shape[0] < 5:
                im = im.transpose([1, 2, 0])
            wandb_images.append(wandb.Image(im))
        wandb.log({tag: wandb_images}, step=global_step)

    @multi_process_safe
    def add_video(self, tag: str, video: Union[torch.Tensor, np.array], global_step: int = 0):
        super().add_video(tag, video, global_step)

        if video.ndim > 4:
            for index, vid in enumerate(video):
                self.add_video(tag=f"{tag}_{index}", video=vid, global_step=global_step)
        else:
            if isinstance(video, torch.Tensor):
                video = video.cpu().detach().numpy()
            wandb.log({tag: wandb.Video(video, fps=4)}, step=global_step)

    @multi_process_safe
    def add_histogram(self, tag: str, values: Union[torch.Tensor, np.array], bins: str, global_step: int = 0):
        super().add_histogram(tag, values, bins, global_step)
        wandb.log({tag: wandb.Histogram(values, num_bins=bins)}, step=global_step)

    @multi_process_safe
    def add_text(self, tag: str, text_string: str, global_step: int = 0):
        super().add_text(tag, text_string, global_step)
        wandb.log({tag: text_string}, step=global_step)

    @multi_process_safe
    def add_figure(self, tag: str, figure: plt.figure, global_step: int = 0):
        super().add_figure(tag, figure, global_step)
        wandb.log({tag: figure}, step=global_step)

    @multi_process_safe
    def close(self):
        super().close()
        wandb.finish()

    @multi_process_safe
    def add_file(self, file_name: str = None):
        super().add_file(file_name)
        wandb.save(glob_str=os.path.join(self._local_dir, file_name), base_path=self._local_dir, policy="now")

    @multi_process_safe
    def upload(self):
        super().upload()

        if self.save_tensorboard_wandb:
            wandb.save(glob_str=self._get_tensorboard_file_name(), base_path=self._local_dir, policy="now")

        if self.save_logs_wandb:
            wandb.save(glob_str=self.experiment_log_path, base_path=self._local_dir, policy="now")

    def _save_wandb_artifact(self, path):
        """Upload a file or a directory as a Weights & Biases Artifact.
        Note that this function can be called only after wandb.init()

        :param path: the local full path to the pth file to be uploaded
        """
        artifact = wandb.Artifact(f"{wandb.run.id}-checkpoint", type="model")
        if os.path.isdir(path):
            artifact.add_dir(path)
        elif os.path.isfile(path):
            artifact.add_file(path)
        wandb.log_artifact(artifact)

    @multi_process_safe
    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = 0):
        name = f"ckpt_{global_step}.pth" if tag is None else tag
        if not name.endswith(".pth"):
            name += ".pth"

        path = os.path.join(self._local_dir, name)
        torch.save(state_dict, path)

        if self.save_checkpoints_wandb:
            if self.s3_location_available:
                self.model_checkpoints_data_interface.save_remote_checkpoints_file(self.experiment_name, self._local_dir, name)
            if self.save_checkpoint_as_artifact:
                self._save_wandb_artifact(path)
            else:
                wandb.save(glob_str=path, base_path=self._local_dir, policy="now")

    def _get_tensorboard_file_name(self):
        try:
            tb_file_path = self.tensorboard_writer.file_writer.event_writer._file_name
        except RuntimeError:
            logger.warning("tensorboard file could not be located for ")
            return None

        return tb_file_path

    def _get_wandb_id(self):
        for file in os.listdir(self._local_dir):
            if file.startswith(WANDB_ID_PREFIX):
                return file.replace(WANDB_ID_PREFIX, "")

    def _set_wandb_id(self, id):
        for file in os.listdir(self._local_dir):
            if file.startswith(WANDB_ID_PREFIX):
                os.remove(os.path.join(self._local_dir, file))

        os.mknod(os.path.join(self._local_dir, f"{WANDB_ID_PREFIX}{id}"))

    def add(self, tag: str, obj: Any, global_step: int = None):
        pass

    def _get_include_paths(self):
        """
        Look for .wandbinclude file in parent dirs and return the list of paths defined in the file.

        file structure is a single relative (i.e. src/) or a single type (i.e *.py)in each line.
        the paths and types in the file are the paths and types to be included in code upload to wandb
        :return: if file exists, return the list of paths and a list of types defined in the file
        """

        wandb_include_file_path = self._search_upwards_for_file(WANDB_INCLUDE_FILE_NAME)
        if wandb_include_file_path is not None:
            with open(wandb_include_file_path) as file:
                lines = file.readlines()

            base_path = os.path.dirname(wandb_include_file_path)
            paths = []
            types = []
            for line in lines:
                line = line.strip().strip("/n")
                if line == "" or line.startswith("#"):
                    continue

                if line.startswith("*."):
                    types.append(line.replace("*", ""))
                else:
                    paths.append(os.path.join(base_path, line))
            return base_path, paths, types

        return ".", [], []

    @staticmethod
    def _search_upwards_for_file(file_name: str):
        """
        Search in the current directory and all directories above it for a file of a particular name.
        :param file_name: file name to look for.
        :return: pathlib.Path, the location of the first file found or None, if none was found
        """

        try:
            cur_dir = os.getcwd()
            while cur_dir != "/":
                if file_name in os.listdir(cur_dir):
                    return os.path.join(cur_dir, file_name)
                else:
                    cur_dir = os.path.dirname(cur_dir)
        except RuntimeError:
            return None

        return None

    def download_remote_ckpt(self, *args, **kwargs):
        wandb.restore("ckpt_latest.pth", replace=True, root=self.local_dir())
