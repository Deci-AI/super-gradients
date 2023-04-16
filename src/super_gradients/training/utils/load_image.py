from typing import Union, List
import PIL

import numpy as np
import torch
import requests
from urllib.parse import urlparse

IMG_EXTENSIONS = ("bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm")
ImageType = Union[str, np.ndarray, torch.Tensor, PIL.Image.Image]


def load_images(images: Union[List[ImageType], ImageType]) -> List[np.ndarray]:
    """Load a single image or a list of images and return them as a list of numpy arrays.

    Supported image types include:
        - numpy.ndarray:    A numpy array representing the image
        - torch.Tensor:     A PyTorch tensor representing the image
        - PIL.Image.Image:  A PIL Image object
        - str:              A string representing either a local file path or a URL to an image

    :param images:  Single image or a list of images of supported types.
    :return:        List of images as numpy arrays. If loaded from string, the image will be returned as RGB.
    """
    if isinstance(images, list):
        return [load_image(image=image) for image in images]
    else:
        return [load_image(image=images)]


def load_image(image: ImageType) -> np.ndarray:
    """Load a single image and return it as a numpy arrays.

    Supported image types include:
        - numpy.ndarray:    A numpy array representing the image
        - torch.Tensor:     A PyTorch tensor representing the image
        - PIL.Image.Image:  A PIL Image object
        - str:              A string representing either a local file path or a URL to an image

    :param image: Single image of supported types.
    :return:      Image as numpy arrays. If loaded from string, the image will be returned as RGB.
    """
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, torch.Tensor):
        return image.numpy()
    elif isinstance(image, PIL.Image.Image):
        return load_np_image_from_pil(image)
    elif isinstance(image, str):
        image = load_pil_image_from_str(image_str=image)
        return load_np_image_from_pil(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def load_np_image_from_pil(image: PIL.Image.Image) -> np.ndarray:
    """Convert a PIL image to numpy array in RGB format."""
    return np.asarray(image.convert("RGB"))


def load_pil_image_from_str(image_str: str) -> PIL.Image.Image:
    """Load an image based on a string (local file path or URL)."""

    if is_url(image_str):
        response = requests.get(image_str, stream=True)
        response.raise_for_status()
        return PIL.Image.open(response.raw)
    else:
        return PIL.Image.open(image_str)


def is_url(url: str) -> bool:
    """Check if the given string is a URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except Exception:
        return False


def is_image(filename: str) -> bool:
    """Check if the given file name refers to image.

    :param filename:    The filename to check.
    :return:            True if the file is an image, False otherwise.
    """
    return filename.split(".")[-1].lower() in IMG_EXTENSIONS
