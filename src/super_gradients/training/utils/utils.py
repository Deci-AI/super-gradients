import collections
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union, List, Dict, Any
from zipfile import ZipFile
import os
from jsonschema import validate
import tarfile
from PIL import Image, ExifTags
import re
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
            raise AttributeError("schema was not set")
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
        self.on_gpu = device == "cuda"
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
        return ((self._sum / self._count).__float__()) if self._sum.dim() < 1 else tuple((self._sum / self._count).cpu().numpy())

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


def fuzzy_keys(params: Mapping) -> List[str]:
    """
    Returns params.key() removing leading and trailing white space, lower-casing and dropping symbols.
    :param params: Mapping, the mapping containing the keys to be returned.
    :return: List[str], list of keys as discussed above.
    """
    return [fuzzy_str(s) for s in params.keys()]


def fuzzy_str(s: str):
    """
    Returns s removing leading and trailing white space, lower-casing and drops
    :param s: str, string to apply the manipulation discussed above.
    :return: str, s after the manipulation discussed above.
    """
    return re.sub(r"[^\w]", "", s).replace("_", "").lower()


def _get_fuzzy_attr_map(params):
    return {fuzzy_str(a): a for a in params.__dir__()}


def _has_fuzzy_attr(params, name):
    return fuzzy_str(name) in _get_fuzzy_attr_map(params)


def get_fuzzy_mapping_param(name: str, params: Mapping):
    """
    Returns parameter value, with key=name with no sensitivity to lowercase, uppercase and symbols.
    :param name: str, the key in params which is fuzzy-matched and retruned.
    :param params: Mapping, the mapping containing param.
    :return:
    """
    fuzzy_params = {fuzzy_str(key): params[key] for key in params.keys()}
    return fuzzy_params[fuzzy_str(name)]


def get_fuzzy_attr(params: Any, name: str):
    """
    Returns attribute (same functionality as getattr), but non sensitive to symbols, uppercase and lowercase.
    :param params: Any, any object which wed looking for the attribute name in.
    :param name: str, the attribute of param to be returned.
    :return: Any, the attribute value or None when not fuzzy matching of the attribute is found
    """
    return getattr(params, _get_fuzzy_attr_map(params)[fuzzy_str(name)])


def fuzzy_idx_in_list(name: str, lst: List[str]) -> int:
    """
    Returns the index of name in lst, with non sensitivity to symbols, uppercase and lowercase.
    :param name: str, the name to be searched in lst.
    :param lst: List[str], the list as described above.
    :return: int, index of name in lst in the matter discussed above.
    """
    return [fuzzy_str(x) for x in lst].index(fuzzy_str(name))


def get_param(params, name, default_val=None):
    """
    Retrieves a param from a parameter object/dict . If the parameter does not exist, will return default_val.
    In case the default_val is of type dictionary, and a value is found in the params - the function
    will return the default value dictionary with internal values overridden by the found value
    IMPORTANT: Not sensitive to lowercase, uppercase and symbols.

    i.e.
    default_opt_params = {'lr':0.1, 'momentum':0.99, 'alpha':0.001}
    training_params = {'optimizer_params': {'lr':0.0001}, 'batch': 32 .... }
    get_param(training_params, name='OptimizerParams', default_val=default_opt_params)
    will return {'lr':0.0001, 'momentum':0.99, 'alpha':0.001}


    :param params:      an object (typically HpmStruct) or a dict holding the params
    :param name:        name of the searched parameter
    :param default_val: assumed to be the same type as the value searched in the params
    :return:            the found value, or default if not found
    """
    if isinstance(params, Mapping):
        if name in params:
            param_val = params[name]

        elif fuzzy_str(name) in fuzzy_keys(params):
            param_val = get_fuzzy_mapping_param(name, params)

        else:
            param_val = default_val
    elif hasattr(params, name):
        param_val = getattr(params, name)
    elif _has_fuzzy_attr(params, name):
        param_val = get_fuzzy_attr(params, name)
    else:
        param_val = default_val

    if isinstance(default_val, Mapping):
        return {**default_val, **param_val}
    else:
        return param_val


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
    rank = 0 if not is_ddp else int(device.split(":")[1])
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
    frameworks_dict = {
        "TENSORFLOW1": ".pb",
        "TENSORFLOW2": ".zip",
        "PYTORCH": ".pth",
        "ONNX": ".onnx",
        "TENSORRT": ".pkl",
        "OPENVINO": ".pkl",
        "TORCHSCRIPT": ".pth",
        "TVM": "",
        "KERAS": ".h5",
        "TFLITE": ".tflite",
    }

    if framework.upper() not in frameworks_dict.keys():
        raise ValueError(f"Unsupported framework: {framework}")

    return frameworks_dict[framework.upper()]


def check_models_have_same_weights(model_1: torch.nn.Module, model_2: torch.nn.Module):
    """
    Checks whether two networks have the same weights

    @param model_1: Net to be checked
    @param model_2: Net to be checked
    @return: True iff the two networks have the same weights
    """
    model_1, model_2 = model_1.to("cpu"), model_2.to("cpu")
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print(f"Layer names match but layers have different weights for layers: {key_item_1[0]}")
    if models_differ == 0:
        return True
    else:
        return False


def recursive_override(base: dict, extension: dict):
    for k, v in extension.items():
        if k in base:
            if isinstance(v, Mapping) and isinstance(base[k], Mapping):
                recursive_override(base[k], extension[k])
            else:
                base[k] = extension[k]
        else:
            base[k] = extension[k]


def download_and_unzip_from_url(url, dir=".", unzip=True, delete=True):
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
            print(f"Downloading {url} to {f}...")
            torch.hub.download_url_to_file(url, f, progress=True)  # torch download
        if unzip and f.suffix in (".zip", ".gz"):
            print(f"Unzipping {f}...")
            if f.suffix == ".zip":
                ZipFile(f).extractall(path=dir)  # unzip
            elif f.suffix == ".gz":
                os.system(f"tar xfz {f} --directory {f.parent}")  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    for u in [url] if isinstance(url, (str, Path)) else url:
        download_one(u, dir)


def download_and_untar_from_url(urls: List[str], dir: Union[str, Path] = "."):
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
            logger.info(f"Downloading {url} to {filepath}...")
            torch.hub.download_url_to_file(url, str(filepath), progress=True)

        modes = {".tar.gz": "r:gz", ".tar": "r:"}
        assert filepath.suffix in modes.keys(), f"{filepath} has {filepath.suffix} suffix which is not supported"

        logger.info(f"Extracting to {dir}...")
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
        if value == "Orientation":
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
        print("Caught Exception trying to rotate: " + str(image) + str(ex))
    width, height = image_size
    return height, width


def get_image_size_from_path(img_path: str) -> Tuple[int, int]:
    """Get the image size of an image at a specific path"""
    with open(img_path, "rb") as f:
        return exif_size(Image.open(f))


def override_default_params_without_nones(params: Dict, default_params: Mapping) -> Dict:
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


def default_collate(batch):
    r"""
    Function that takes in a batch of data and puts the elements within the batch
    into a tensor with an additional outer dimension - batch size. The exact output type can be
    a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
    Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
    This is used as the default function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

    Here is the general input type (based on the type of the element within the batch) to output type mapping:

        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]),
          default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]),
          default_collate([V2_1, V2_2, ...]), ...]`

    Args:
        batch: a single batch to be collated

    Examples:
        >>> # Example with a batch of `int`s:
        >>> default_collate([0, 1, 2, 3])
        tensor([0, 1, 2, 3])
        >>> # Example with a batch of `str`s:
        >>> default_collate(['a', 'b', 'c'])
        ['a', 'b', 'c']
        >>> # Example with `Map` inside the batch:
        >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
        {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
        >>> # Example with `NamedTuple` inside the batch:
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> default_collate([Point(0, 0), Point(1, 1)])
        Point(x=tensor([0, 1]), y=tensor([0, 1]))
        >>> # Example with `Tuple` inside the batch:
        >>> default_collate([(0, 1), (2, 3)])
        [tensor([0, 2]), tensor([1, 3])]
        >>> # Example with `List` inside the batch:
        >>> default_collate([[0, 1], [2, 3]])
        [tensor([0, 2]), tensor([1, 3])]
    """
    np_str_obj_array_pattern = re.compile(r"[SaUO]")
    default_collate_err_msg_format = "default_collate: batch must contain tensors, numpy arrays, numbers, " "dicts or lists; found {}"
    string_classes = (str, bytes)

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [default_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
