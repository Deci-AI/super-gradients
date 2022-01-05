from abc import abstractmethod, ABC
from typing import Union, Any

import numpy as np
from PIL import Image
import torch


class AbstractSGLogger(ABC):
    """
    A SGLogger handles all outputs of the training process.
    Every generated file, log, metrics value, image or other artifacts produced by the trainer will be processed and saved.

    Inheriting SGLogger can be used in order to integrate experiment management framework, special storage setting, a specific logging library etc.

    Important: The BaseSGLogger class (inheriting from SGLogger) is used by the trainer by default. When defining your own SGLogger you will
    override all default output functionality. No files will saved to disk and no data will be collected.
    Make sure you either implement this functionality or use SGLoggers.Compose([BaseSGLogger(...), YourSGLogger(...)]) to build on top of it.
    """

    @abstractmethod
    def add(self, tag: str, obj: Any, global_step: int = None):
        """
        A generic function for adding any type of data to the SGLogger. By default, this function is not called by the SGModel, BaseSGLogger
        does nothing with this type of data. But if you need to pass a data type which is not supported by any of the following abstract methods, use this
        method.
        """
        raise NotImplementedError

    @abstractmethod
    def add_config(self, tag: str, config: dict):
        """
        Add the configuration (settings and hyperparameters) to the SGLoggers.
        Typically, this function will add the configuration dictionary to logs,
        write it to tensorboard, send it to an experiment management framework ect.

        :param tag: Data identifier
        :param config: a dictionary of the experiment config
        """
        raise NotImplementedError

    @abstractmethod
    def add_scalar(self, tag: str, scalar_value: float, global_step: int = None):
        """
        Add scalar data to SGLogger.
        Typically, this function will add scalar to tensorboard or other experiment management framework.

        :param tag: Data identifier
        :param scalar_value: Value to save
        :param global_step: Global step value to record
        """
        raise NotImplementedError

    @abstractmethod
    def add_scalars(self, tag_scalar_dict: dict, global_step: int = None):
        """
        Adds multiple scalar data to SGLogger.
        Typically, this function will add scalars to tensorboard or other experiment management framework.

        :param tag_scalar_dict: a dictionary {tag(str): value(float)} of the scalars.
        :param global_step: Global step value to record
        """
        raise NotImplementedError

    @abstractmethod
    def add_image(self, tag: str, image: Union[torch.Tensor, np.array, Image.Image], data_format: str = 'CHW', global_step: int = None):
        """
        Add a single image to SGLogger.
        Typically, this function will add an image to tensorboard, save it to disk or add it to experiment management framework.

        :param tag: Data identifier
        :param image: an image to be added. The values should lie in [0, 255] for type uint8 or [0, 1] for type float.
        :param data_format: Image data format specification of the form CHW, HWC, HW, WH, etc.
        :param global_step: Global step value to record
        """
        raise NotImplementedError

    @abstractmethod
    def add_images(self, tag: str, images: Union[torch.Tensor, np.array], data_format='NCHW', global_step: int = None):
        """
        Add multiple images to SGLogger.
        Typically, this function will add images to tensorboard, save them to disk or add them to experiment management framework.

        :param tag: Data identifier
        :param images: images to be added. The values should lie in [0, 255] for type uint8 or [0, 1] for type float.
        :param data_format: Image data format specification of the form NCHW, NHWC, NHW, NWH, etc.
        :param global_step: Global step value to record
        """
        raise NotImplementedError

    @abstractmethod
    def add_histogram(self, tag: str, values: Union[torch.Tensor, np.array], bins: Union[str, np.array, list, int] = 'auto', global_step: int = None):
        """
        Add a histogram to SGLogger.
        Typically, this function will add a histogram to tensorboard or add it to experiment management framework.

        :param tag: Data identifier
        :param values: Values to build histogram
        :param bins: This determines how the bins are made.
            If bins is an int, it defines the number of equal-width bins in the given range
            If bins is a sequence, it defines a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform bin widths.
            If bins is a string, it defines the method used to calculate the optimal bin width, as defined by
            https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
            one of [‘sqrt’, ’auto’, ‘fd’, ‘doane’, ‘scott’, ‘stone’...]
        :param global_step: Global step value to record
        """
        raise NotImplementedError

    @abstractmethod
    def add_text(self, tag: str, text_string: str, global_step: int = None):
        """
        Add a text to SGLogger.
        Typically, this function will add a text to tensorboard or add it to experiment management framework.

        :param tag: Data identifier
        :param text_string: the text to be added
        :param global_step: Global step value to record
        """
        raise NotImplementedError

    @abstractmethod
    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = None):
        """
        Add a checkpoint to SGLogger
        Typically, this function will write a torch file to disk, upload it to remote storage or to experiment management framework.

        :param tag: Data identifier
        :param state_dict: the state dict to save. The state dict includes more than just the model weight and may include any of:
                net: model weights
                acc: current accuracy (depends on metrics)
                epoch: current epoch
                optimizer_state_dict: optimizer state
                scaler_state_dict: torch.amp.scaler sate
        :param global_step: Global step value to record
        """
        raise NotImplementedError

    @abstractmethod
    def add_file(self, file_name: str = None):
        """
        Add a file from the checkpoint directory to the logger (usually, upload the file or adds it to an artifact)
        """
        raise NotImplementedError

    @abstractmethod
    def upload(self):
        """
        Upload any files which should be stored on remote storage
        """
        raise NotImplementedError

    @abstractmethod
    def flush(self):
        """
        Flush the SGLogger's cache
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """
        Close the SGLogger
        """
        raise NotImplementedError

    @abstractmethod
    def local_dir(self) -> str:
        """
        A getter for the full/absolute path where all files are saved locally
        :return:
        """
        raise NotImplementedError
