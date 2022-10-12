import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union, List, Dict
from zipfile import ZipFile
import os
from jsonschema import validate
import tarfile
from PIL import Image, ExifTags

import torch
import torch.nn as nn


# These functions changed from torch 1.2 to torch 1.3

import random
import numpy as np
from importlib import import_module

from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


def empty_list():
    """Instantiate an empty list. This is a workaround to generate a list with a function call in hydra, instead of the "[]"."""
    return list()


def convert_to_tensor(array):
    """Converts numpy arrays and lists to Torch tensors before calculation losses
    :param array: torch.tensor / Numpy array / List
    """
    return torch.FloatTensor(array) if type(array) != torch.Tensor else array


class HpmStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.schema = None

    def set_schema(self, schema: dict):
        self.schema = schema

    def override(self, **entries):
        recursive_override(self.__dict__, entries)

    def to_dict(self, include_schema=True) -> dict:
        """Convert this HpmStruct instance into a dict.
        :param include_schema: If True, also return the field "schema"
        :return: Dict representation of this HpmStruct instance.
        """
        out_dict = self.__dict__.copy()
        if not include_schema:
            out_dict.pop("schema")
        return out_dict

    def validate(self):
        """
        Validate the current dict values according to the provided schema
        :raises
            `AttributeError` if schema was not set
            `jsonschema.exceptions.ValidationError` if the instance is invalid
            `jsonschema.exceptions.SchemaError` if the schema itselfis invalid
        """
        if self.schema is None:
            raise AttributeError('schema was not set')
        else:
            validate(self.__dict__, self.schema)


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module  # that I actually define.

    def forward(self, x):
        return self.module(x)


class Timer:
    """A class to measure time handling both GPU & CPU processes
    Returns time in milliseconds"""

    def __init__(self, device: str):
        """
        :param device: str
            'cpu'\'cuda'
        """
        self.on_gpu = (device == 'cuda')
        # On GPU time is measured using cuda.events
        if self.on_gpu:
            self.starter = torch.cuda.Event(enable_timing=True)
            self.ender = torch.cuda.Event(enable_timing=True)
        # On CPU time is measured using time
        else:
            self.starter, self.ender = 0, 0

    def start(self):
        if self.on_gpu:
            self.starter.record()
        else:
            self.starter = time.time()

    def stop(self):
        if self.on_gpu:
            self.ender.record()
            torch.cuda.synchronize()
            timer = self.starter.elapsed_time(self.ender)
        else:
            # Time measures in seconds -> convert to milliseconds
            timer = (time.time() - self.starter) * 1000

        # Return time in milliseconds
        return timer


class AverageMeter:
    """A class to calculate the average of a metric, for each batch
    during training/testing"""

    def __init__(self):
        self._sum = None
        self._count = 0

    def update(self, value: Union[float, tuple, list, torch.Tensor], batch_size: int):

        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        if self._sum is None:
            self._sum = value * batch_size
        else:
            self._sum += value * batch_size

        self._count += batch_size

    @property
    def average(self):
        if self._sum is None:
            return 0
        return ((self._sum / self._count).__float__()) if self._sum.dim() < 1 else tuple(
            (self._sum / self._count).cpu().numpy())

        # return (self._sum / self._count).__float__() if self._sum.dim() < 1 or len(self._sum) == 1 \
        #     else tuple((self._sum / self._count).cpu().numpy())


def tensor_container_to_device(obj: Union[torch.Tensor, tuple, list, dict], device: str, non_blocking=True):
    """
    recursively send compounded objects to device (sending all tensors to device and maintaining structure)
        :param obj           the object to send to device (list / tuple / tensor / dict)
        :param device:       device to send the tensors to
        :param non_blocking: used for DistributedDataParallel
        :returns        an object with the same structure (tensors, lists, tuples) with the device pointers (like
                        the return value of Tensor.to(device)
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    elif isinstance(obj, tuple):
        return tuple(tensor_container_to_device(x, device, non_blocking=non_blocking) for x in obj)
    elif isinstance(obj, list):
        return [tensor_container_to_device(x, device, non_blocking=non_blocking) for x in obj]
    elif isinstance(obj, dict):
        return {k: tensor_container_to_device(v, device, non_blocking=non_blocking) for k, v in obj.items()}
    else:
        return obj


def get_param(params, name, default_val=None):
    """
    Retrieves a param from a parameter object/dict. If the parameter does not exist, will return default_val.
    In case the default_val is of type dictionary, and a value is found in the params - the function
    will return the default value dictionary with internal values overridden by the found value

    i.e.
    default_opt_params = {'lr':0.1, 'momentum':0.99, 'alpha':0.001}
    training_params = {'optimizer_params': {'lr':0.0001}, 'batch': 32 .... }
    get_param(training_params, name='optimizer_params', default_val=default_opt_params)
    will return {'lr':0.0001, 'momentum':0.99, 'alpha':0.001}

    :param params:      an object (typically HpmStruct) or a dict holding the params
    :param name:        name of the searched parameter
    :param default_val: assumed to be the same type as the value searched in the params
    :return:            the found value, or default if not found
    """
    if isinstance(params, dict):
        if name in params:
            if isinstance(default_val, dict):
                return {**default_val, **params[name]}
            else:
                return params[name]
        else:
            return default_val
    elif hasattr(params, name):
        if isinstance(default_val, dict):
            return {**default_val, **getattr(params, name)}
        else:
            return getattr(params, name)
    else:
        return default_val


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@static_vars(printed=set())
def print_once(s: str):
    if s not in print_once.printed:
        print_once.printed.add(s)
        print(s)


def move_state_dict_to_device(model_sd, device):
    """
    Moving model state dict tensors to target device (cuda or cpu)
    :param model_sd: model state dict
    :param device: either cuda or cpu
    """
    for k, v in model_sd.items():
        model_sd[k] = v.to(device)
    return model_sd


def random_seed(is_ddp, device, seed):
    """
    Sets random seed of numpy, torch and random.

    When using ddp a seed will be set for each process according to its local rank derived from the device number.
    :param is_ddp: bool, will set different random seed for each process when using ddp.
    :param device: 'cuda','cpu', 'cuda:<device_number>'
    :param seed: int, random seed to be set
    """
    rank = 0 if not is_ddp else int(device.split(':')[1])
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def load_func(dotpath: str):
    """
    load function in module.  function is right-most segment.

    Used for passing functions (without calling them) in yaml files.

    @param dotpath: path to module.
    @return: a python function
    """
    module_, func = dotpath.rsplit(".", maxsplit=1)
    m = import_module(module_)
    return getattr(m, func)


def get_filename_suffix_by_framework(framework: str):
    """
    Return the file extension of framework.

    @param framework: (str)
    @return: (str) the suffix for the specific framework
    """
    frameworks_dict = \
        {
            'TENSORFLOW1': '.pb',
            'TENSORFLOW2': '.zip',
            'PYTORCH': '.pth',
            'ONNX': '.onnx',
            'TENSORRT': '.pkl',
            'OPENVINO': '.pkl',
            'TORCHSCRIPT': '.pth',
            'TVM': '',
            'KERAS': '.h5',
            'TFLITE': '.tflite'
        }

    if framework.upper() not in frameworks_dict.keys():
        raise ValueError(f'Unsupported framework: {framework}')

    return frameworks_dict[framework.upper()]


def check_models_have_same_weights(model_1: torch.nn.Module, model_2: torch.nn.Module):
    """
    Checks whether two networks have the same weights

    @param model_1: Net to be checked
    @param model_2: Net to be checked
    @return: True iff the two networks have the same weights
    """
    model_1, model_2 = model_1.to('cpu'), model_2.to('cpu')
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print(f'Layer names match but layers have different weights for layers: {key_item_1[0]}')
    if models_differ == 0:
        return True
    else:
        return False


def recursive_override(base: dict, extension: dict):
    for k, v in extension.items():
        if k in base:
            if isinstance(v, Mapping):
                recursive_override(base[k], extension[k])
            else:
                base[k] = extension[k]
        else:
            base[k] = extension[k]


def download_and_unzip_from_url(url, dir='.', unzip=True, delete=True):
    """
    Downloads a zip file from url to dir, and unzips it.

    :param url: Url to download the file from.
    :param dir: Destination directory.
    :param unzip: Whether to unzip the downloaded file.
    :param delete: Whether to delete the zip file.

    used to downlaod VOC.

    Source:
    https://github.com/ultralytics/yolov5/blob/master/data/VOC.yaml
    """

    def download_one(url, dir):
        # Download 1 file
        f = dir / Path(url).name  # filename
        if Path(url).is_file():  # exists in current path
            Path(url).rename(f)  # move to dir
        elif not f.exists():
            print(f'Downloading {url} to {f}...')
            torch.hub.download_url_to_file(url, f, progress=True)  # torch download
        if unzip and f.suffix in ('.zip', '.gz'):
            print(f'Unzipping {f}...')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)  # unzip
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent}')  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    for u in [url] if isinstance(url, (str, Path)) else url:
        download_one(u, dir)


def download_and_untar_from_url(urls: List[str], dir: Union[str, Path] = '.'):
    """
    Download a file from url and untar.

    :param urls:    Url to download the file from.
    :param dir:     Destination directory.
    """
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)

    for url in urls:
        url_path = Path(url)
        filepath = dir / url_path.name

        if url_path.is_file():
            url_path.rename(filepath)
        elif not filepath.exists():
            logger.info(f'Downloading {url} to {filepath}...')
            torch.hub.download_url_to_file(url, str(filepath), progress=True)

        modes = {".tar.gz": "r:gz", ".tar": "r:"}
        assert filepath.suffix in modes.keys(), f"{filepath} has {filepath.suffix} suffix which is not supported"

        logger.info(f'Extracting to {dir}...')
        with tarfile.open(filepath, mode=modes[filepath.suffix]) as f:
            f.extractall(dir)
        filepath.unlink()


def make_divisible(x: int, divisor: int, ceil: bool = True) -> int:
    """
    Returns x evenly divisible by divisor.
    If ceil=True it will return the closest larger number to the original x, and ceil=False the closest smaller number.
    """
    if ceil:
        return math.ceil(x / divisor) * divisor
    else:
        return math.floor(x / divisor) * divisor


def check_img_size_divisibility(img_size: int, stride: int = 32) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """
    :param img_size: Int, the size of the image (H or W).
    :param stride: Int, the number to check if img_size is divisible by.
    :return: (True, None) if img_size is divisble by stride, (False, Suggestions) if it's not.
        Note: Suggestions are the two closest numbers to img_size that *are* divisible by stride.
        For example if img_size=321, stride=32, it will return (False,(352, 320)).
    """
    new_size = make_divisible(img_size, int(stride))
    if new_size != img_size:
        return False, (new_size, make_divisible(img_size, int(stride), ceil=False))
    else:
        return True, None


@lru_cache(None)
def get_orientation_key() -> int:
    """Get the orientation key according to PIL, which is useful to get the image size for instance
    :return: Orientation key according to PIL"""
    for key, value in ExifTags.TAGS.items():
        if value == 'Orientation':
            return key


def exif_size(image: Image) -> Tuple[int, int]:
    """Get the size of image.
    :param image:   The image to get size from
    :return:        (height, width)
    """

    orientation_key = get_orientation_key()

    image_size = image.size
    try:
        exif_data = image._getexif()
        if exif_data is not None:
            rotation = dict(exif_data.items())[orientation_key]
            # ROTATION 270
            if rotation == 6:
                image_size = (image_size[1], image_size[0])
            # ROTATION 90
            elif rotation == 8:
                image_size = (image_size[1], image_size[0])
    except Exception as ex:
        print('Caught Exception trying to rotate: ' + str(image) + str(ex))
    width, height = image_size
    return height, width


def get_image_size_from_path(img_path: str) -> Tuple[int, int]:
    """Get the image size of an image at a specific path"""
    with open(img_path, 'rb') as f:
        return exif_size(Image.open(f))


def override_default_params_without_nones(params: Dict, default_params: Dict) -> Dict:
    """
    Helper method for overriding default dictionary's entries excluding entries with None values.
    :param params: dict, output dictionary which will take the defaults.
    :param default_params: dict, dictionary for the defaults.
    :return: dict, params after manipulation,
    """
    for key, val in default_params.items():
        if key not in params.keys() or params[key] is None:
            params[key] = val
    return params
