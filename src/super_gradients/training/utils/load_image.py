from typing import Union, List, Iterable
import PIL

import os
from PIL import Image

import numpy as np
import torch
import requests
from urllib.parse import urlparse

from super_gradients.training.utils.utils import generate_batch

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
    return [image for image in generate_loaded_image(images=images)]


def generate_loaded_image(images: Union[List[ImageType], ImageType]) -> Iterable[np.ndarray]:
    if isinstance(images, list):
        for image in images:
            yield load_image(image=image)
    else:
        yield load_image(image=images)


def generate_loaded_image_batch(images: Union[List[ImageType], ImageType], batch_size: int) -> List[np.ndarray]:
    images_generator = generate_loaded_image(images=images)
    yield from generate_batch(iterable=images_generator, batch_size=batch_size)


def list_images_in_folder(directory: str) -> List[str]:
    """List all the images in a directory.
    :param directory: The path to the directory containing the images.
    :return: A list of image file names.
    """
    files = os.listdir(directory)
    images_paths = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    return images_paths


# def load_directory_images_iterator(directory: str, limit: Optional[int] = None) -> Iterator[Image.Image]:
#     """Return an iterator that loads each image one by one with tqdm.
#
#     :param directory:   The path to the directory containing the images.
#     :param limit:       The maximum number of images to load
#     :return: An iterator object that yields loaded images.
#     """
#     images_paths = list_images_in_folder(directory)
#     if limit is not None:
#         images_paths = images_paths[:limit]
#
#     for image_path in images_paths:
#         yield load_image(image=image_path)


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


def save_image(image: np.ndarray, path: str) -> None:
    Image.fromarray(image).save(path)


def is_url(url: str) -> bool:
    """Check if the given string is a URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except Exception:
        return False
