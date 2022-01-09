import json
import os
import time
import signal

from typing import Union, Any

import psutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

from super_gradients.common import ADNNModelRepositoryDataInterfaces
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.sg_loggers.abstract_sg_logger import AbstractSGLogger
from super_gradients.common.environment.env_helpers import multi_process_safe
from super_gradients.training.utils import sg_model_utils
from super_gradients.training.params import TrainingParams

logger = get_logger(__name__)


class BaseSGLogger(AbstractSGLogger):

    def __init__(self, project_name: str,
                 experiment_name: str,
                 storage_location: str,
                 resumed: bool,
                 training_params: TrainingParams,
                 checkpoints_dir_path: str,
                 tb_files_user_prompt: bool = False,
                 launch_tensorboard: bool = False,
                 tensorboard_port: int = None,
                 save_checkpoints_remote: bool = True,
                 save_tensorboard_remote: bool = True,
                 save_logs_remote: bool = True):
        """

        :param experiment_name: Used for logging and loading purposes
        :param storage_location: If set to 's3' (i.e. s3://my-bucket) saves the Checkpoints in AWS S3 otherwise saves the Checkpoints Locally
        :param resumed: if true, then old tensorboard files will *not* be deleted when tb_files_user_prompt=True
        :param training_params: training_params for the experiment.
        :param checkpoints_dir_path: Local root directory path where all experiment logging directories will
                                                 reside.
        :param tb_files_user_prompt: Asks user for Tensorboard deletion prompt.
        :param launch_tensorboard: Whether to launch a TensorBoard process.
        :param tensorboard_port: Specific port number for the tensorboard to use when launched (when set to None, some free port
                    number will be used
        :param save_checkpoints_remote: Saves checkpoints in s3.
        :param save_tensorboard_remote: Saves tensorboard in s3.
        :param save_logs_remote: Saves log files in s3.
        """
        super().__init__()
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.storage_location = storage_location

        if storage_location.startswith('s3'):
            self.save_checkpoints_remote = save_checkpoints_remote
            self.save_tensorboard_remote = save_tensorboard_remote
            self.save_logs_remote = save_logs_remote
            self.remote_storage_available = True
        else:
            self.remote_storage_available = False
            if save_checkpoints_remote:
                logger.error('save_checkpoints_remote == True but storage_location is not s3 path. Files will not be saved remotely')
            if save_tensorboard_remote:
                logger.error('save_tensorboard_remote == True but storage_location is not s3 path. Files will not be saved remotely')
            if save_logs_remote:
                logger.error('save_logs_remote == True but storage_location is not s3 path. Files will not be saved remotely')

            self.save_checkpoints_remote = False
            self.save_tensorboard_remote = False
            self.save_logs_remote = False

        self.tensor_board_process = None
        self.max_global_steps = training_params.max_epochs
        self._local_dir = checkpoints_dir_path

        self._make_dir()
        self._init_tensorboard(resumed, tb_files_user_prompt)
        self._init_log_file()

        self.model_checkpoints_data_interface = ADNNModelRepositoryDataInterfaces(data_connection_location=self.storage_location)

        if launch_tensorboard:
            self._launch_tensorboard(port=tensorboard_port)

    @multi_process_safe
    def _launch_tensorboard(self, port):
        self.tensor_board_process, _ = sg_model_utils.launch_tensorboard_process(self._local_dir, port=port)

    @multi_process_safe
    def _init_tensorboard(self, resumed, tb_files_user_prompt):
        self.tensorboard_writer = sg_model_utils.init_summary_writer(self._local_dir, resumed, tb_files_user_prompt)

    @multi_process_safe
    def _make_dir(self):
        if not os.path.isdir(self._local_dir):
            os.makedirs(self._local_dir)

    @multi_process_safe
    def _init_log_file(self):
        time_string = time.strftime('%b%d_%H_%M_%S', time.localtime())
        self.log_file_path = f'{self._local_dir}/log_{time_string}.txt'

    @multi_process_safe
    def _write_to_log_file(self, lines: list):
        with open(self.log_file_path, 'a' if os.path.exists(self.log_file_path) else 'w') as log_file:
            for line in lines:
                log_file.write(line + '\n')

    @multi_process_safe
    def add_config(self, tag: str, config: dict):
        log_lines = ['--------- config parameters ----------']
        log_lines.append(json.dumps(config, indent=4, default=str))
        log_lines.append('------- config parameters end --------')

        self.tensorboard_writer.add_text("Hyper_parameters", json.dumps(config, indent=4, default=str).replace(" ", "&nbsp;").replace("\n", "  \n  "))
        self._write_to_log_file(log_lines)

    @multi_process_safe
    def add_scalar(self, tag: str, scalar_value: float, global_step: int = None):
        self.tensorboard_writer.add_scalar(tag=tag.lower().replace(' ', '_'), scalar_value=scalar_value, global_step=global_step)

    @multi_process_safe
    def add_scalars(self, tag_scalar_dict: dict, global_step: int = None):
        """
        add multiple scalars.
        Unlike Tensorboard implementation, this does not add all scalars with a main tag (all scalars to the same chart).
        Instead, scalars are added to tensorboard like in add_scalar and are written in log together.
        """
        for tag, value in tag_scalar_dict.items():
            self.tensorboard_writer.add_scalar(tag=tag.lower().replace(' ', '_'), scalar_value=value, global_step=global_step)

        self.tensorboard_writer.flush()

        # WRITE THE EPOCH RESULTS TO LOG FILE
        log_line = f'\nEpoch ({global_step}/{self.max_global_steps})  - '
        for tag, value in tag_scalar_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            log_line += f'{tag.replace(" ", "_")}: {value}\t'

        self._write_to_log_file([log_line])

    @multi_process_safe
    def add_image(self, tag: str, image: Union[torch.Tensor, np.array, Image.Image], data_format='CHW', global_step: int = None):
        self.tensorboard_writer.add_image(tag=tag, image=image, dataformats=data_format, global_step=global_step)

    @multi_process_safe
    def add_images(self, tag: str, images: Union[torch.Tensor, np.array], data_format='NCHW', global_step: int = None):
        """
        Add multiple images to SGLogger.
        Typically, this function will add a set of images to tensorboard, save them to disk or add it to experiment management framework.

        :param tag: Data identifier
        :param images: images to be added. The values should lie in [0, 255] for type uint8 or [0, 1] for type float.
        :param data_format: Image data format specification of the form NCHW, NHWC, CHW, HWC, HW, WH, etc.
        :param global_step: Global step value to record
        """
        self.tensorboard_writer.add_images(tag=tag, img_tensor=images, dataformats=data_format, global_step=global_step)

    @multi_process_safe
    def add_video(self, tag: str, video: Union[torch.Tensor, np.array], global_step: int = None):
        """
        Add a single video to SGLogger.
        Typically, this function will add a video to tensorboard, save it to disk or add it to experiment management framework.

        :param tag: Data identifier
        :param video: the video to add. shape (N,T,C,H,W) or (T,C,H,W). The values should lie in [0, 255] for type uint8 or [0, 1] for type float.
        :param global_step: Global step value to record
        """
        if video.ndim < 5:
            video = video[None, ]
        self.tensorboard_writer.add_video(tag=tag, video=video, global_step=global_step)

    @multi_process_safe
    def add_histogram(self, tag: str, values: Union[torch.Tensor, np.array], bins: str, global_step: int = None):
        self.tensorboard_writer.add_histogram(tag=tag, values=values, bins=bins, global_step=global_step)

    @multi_process_safe
    def add_model_graph(self, tag: str, model: torch.nn.Module, dummy_input: torch.Tensor):
        """
        Add a pytorch model graph to the SGLogger.
        Only the model structure/architecture will be preserved and collected, NOT the model weights.

        :param tag: Data identifier
        :param model: the model to be added
        :param dummy_input: an input to be used for a forward call on the model
        """
        self.tensorboard_writer.add_graph(model=model, input_to_model=dummy_input)

    @multi_process_safe
    def add_text(self, tag: str, text_string: str, global_step: int = None):
        self.tensorboard_writer.add_text(tag=tag, text_string=text_string, global_step=global_step)

    @multi_process_safe
    def add_figure(self, tag: str, figure: plt.figure, global_step: int = None):
        """
        Add a text to SGLogger.
        Typically, this function will add a figure to tensorboard or add it to experiment management framework.

        :param tag: Data identifier
        :param figure: the figure to add
        :param global_step: Global step value to record
        """
        self.tensorboard_writer.add_figure(tag=tag, figure=figure, global_step=global_step)

    @multi_process_safe
    def add_file(self, file_name: str = None):
        if self.remote_storage_available:
            self.model_checkpoints_data_interface.save_remote_tensorboard_event_files(self.experiment_name, self._local_dir, file_name)

    @multi_process_safe
    def upload(self):
        if self.save_tensorboard_remote:
            self.model_checkpoints_data_interface.save_remote_tensorboard_event_files(self.experiment_name, self._local_dir)

        if self.save_logs_remote:
            log_file_name = self.log_file_path.split('/')[-1]
            self.model_checkpoints_data_interface.save_remote_checkpoints_file(self.experiment_name, self._local_dir, log_file_name)

    @multi_process_safe
    def flush(self):
        self.tensorboard_writer.flush()

    @multi_process_safe
    def close(self):
        self.tensorboard_writer.close()
        if self.tensor_board_process is not None:
            try:
                logger.info('[CLEANUP] - Stopping tensorboard process')
                process = psutil.Process(self.tensor_board_process.pid)
                process.send_signal(signal.SIGTERM)
                logger.info('[CLEANUP] - Successfully stopped tensorboard process')
            except Exception as ex:
                logger.info('[CLEANUP] - Could not stop tensorboard process properly: ' + str(ex))

    @multi_process_safe
    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = None):

        name = f'ckpt_{global_step}.pth' if tag is None else tag
        if not name.endswith('.pth'):
            name += '.pth'

        path = os.path.join(self._local_dir, name)
        torch.save(state_dict, path)

        if self.save_checkpoints_remote:
            self.model_checkpoints_data_interface.save_remote_checkpoints_file(self.experiment_name, self._local_dir, name)

    def add(self, tag: str, obj: Any, global_step: int = None):
        pass

    def local_dir(self) -> str:
        return self._local_dir
