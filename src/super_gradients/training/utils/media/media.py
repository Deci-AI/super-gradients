from typing import Union, List
import os
from PIL import Image
import numpy as np
import torch

from enum import Enum
from super_gradients.training.utils.media.load_image import ImageSource
from super_gradients.training.utils.media.videos import is_video


class MediaType(Enum):
    VIDEO = "video"
    IMAGE_FOLDER = "folder"
    IMAGE = "image"


def determine_media_type(media_files: Union[str, List[str], np.ndarray, Image.Image, torch.Tensor]) -> MediaType:
    """Determine the type of input `media_files`.

    :param media_files: Input to the function. Can be a video, the path of a folder containing images, an image or a list of images.
    :return: An instance of the MediaType Enum indicating the type of input `media_files`.
    """
    # Check if media_files is a video
    if isinstance(media_files, str) and is_video(media_files):
        return MediaType.VIDEO

    # Check if media_files is a path to a folder containing images
    if isinstance(media_files, str) and os.path.isdir(media_files):
        return MediaType.IMAGE_FOLDER

    # Check if media_files is an image
    if isinstance(media_files, ImageSource):
        return MediaType.IMAGE

    # If none of the above conditions are met, raise an exception
    raise ValueError("Invalid input type. Input should be a video, the path of a folder containing images, an image or a list of images.")
