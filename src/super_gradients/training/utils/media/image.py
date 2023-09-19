import warnings
from typing import Union, List, Iterable, Iterator
from typing_extensions import get_args
import PIL
import io
import os
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import torch
import requests
from urllib.parse import urlparse


IMG_EXTENSIONS = ("bmp", "dng", "jpeg", "jpg", "mpo", "pfm", "pgm", "png", "ppm", "tif", "tiff", "webp")
SingleImageSource = Union[str, np.ndarray, torch.Tensor, PIL.Image.Image]
ImageSource = Union[SingleImageSource, List[SingleImageSource], Iterator[SingleImageSource]]


def load_images(images: Union[List[ImageSource], ImageSource]) -> List[np.ndarray]:
    """Load a single image or a list of images and return them as a list of numpy arrays.

    Supported types include:
        - str:              A string representing either an image or an URL.
        - numpy.ndarray:    A numpy array representing the image
        - torch.Tensor:     A PyTorch tensor representing the image
        - PIL.Image.Image:  A PIL Image object
        - List:             A list of images of any of the above types.

    :param images:  Single image or a list of images of supported types.
    :return:        List of images as numpy arrays. If loaded from string, the image will be returned as RGB.
    """
    return [image for image in generate_image_loader(images=images)]


def generate_image_loader(images: Union[List[ImageSource], ImageSource]) -> Iterable[np.ndarray]:
    """Generator that loads images one at a time.

    Supported types include:
        - str:              A string representing either an image or an URL.
        - numpy.ndarray:    A numpy array representing the image
        - torch.Tensor:     A PyTorch tensor representing the image
        - PIL.Image.Image:  A PIL Image object
        - List:             A list of images of any of the above types.

    :param images:  Single image or a list of images of supported types.
    :return:        Generator of images as numpy arrays (H, W, C). If loaded from string, the image will be returned as RGB.
    """
    if isinstance(images, str) and os.path.isdir(images):
        images_paths = list_images_in_folder(images)
        for image_path in images_paths:
            yield load_image(image=image_path)
    elif _is_4d_array(images):
        warnings.warn(
            "It seems you are using predict() with 4D array as input. "
            "Please note we cannot track whether the input was already normalized or not. "
            "You will get incorrect results if you feed batches from train/validation dataloader that were already normalized."
            "Please check https://docs.deci.ai/super-gradients/latest/documentation/source/ModelPredictions.html for more details."
        )
        for image in images:
            yield load_image(image=image)
    elif _is_list_of_images(images=images):
        warnings.warn("It seems you are using predict() with batch input")
        for image in images:
            yield load_image(image=image)
    else:
        yield load_image(image=images)


def _is_4d_array(images: ImageSource) -> bool:
    return isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 4


def _is_list_of_images(images: ImageSource) -> bool:
    return isinstance(images, (list, Iterator))


def list_images_in_folder(directory: str) -> List[str]:
    """List all the images in a directory.
    :param directory: The path to the directory containing the images.
    :return: A list of image file names.
    """
    files = os.listdir(directory)
    images_paths = [os.path.join(directory, f) for f in files if is_image(f)]
    return images_paths


def load_image(image: ImageSource, input_image_channels: int = 3) -> np.ndarray:
    """Load a single image and return it as a numpy arrays (H, W, C).

    Supported image types include:
        - numpy.ndarray:    A numpy array representing the image
        - torch.Tensor:     A PyTorch tensor representing the image
        - PIL.Image.Image:  A PIL Image object
        - str:              A string representing either a local file path or a URL to an image

    :param image: Single image of supported types.
    :param input_image_channels: Number of channels that model expects as input.
                                 This value helps to infer the layout of the input image array.
                                 As of now this argument has default value of 3, but in future it will become mandatory.

    :return:      Image as numpy arrays (H, W, C). If loaded from string, the image will be returned as RGB.
    """
    if isinstance(image, np.ndarray):
        if image.ndim != 3:
            raise ValueError(f"Unsupported image shape: {image.shape}. This function only supports 3-dimensional images.")
        if image.shape[0] == input_image_channels:
            image = np.ascontiguousarray(image.transpose((1, 2, 0)))
        elif image.shape[2] == input_image_channels:
            pass
        else:
            raise ValueError(f"Cannot infer image layout (HWC or CHW) for image of shape {image.shape} while C is {input_image_channels}")

        return image
    elif isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        return load_image(image=image, input_image_channels=input_image_channels)
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
        return PIL.Image.open(io.BytesIO(response.content))
    else:
        return PIL.Image.open(image_str)


def save_image(image: np.ndarray, path: str) -> None:
    """Save a numpy array as an image.
    :param image:  Image to save, (H, W, C), RGB.
    :param path:   Path to save the image to.
    """
    Image.fromarray(image).save(path)


def is_url(url: str) -> bool:
    """Check if the given string is a URL.
    :param url:  String to check.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except Exception:
        return False


def show_image(image: np.ndarray) -> None:
    """Show an image using matplotlib.
    :param image: Image to show in (H, W, C), RGB.
    """
    plt.figure(figsize=(image.shape[1] / 100.0, image.shape[0] / 100.0), dpi=100)
    plt.imshow(image, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def check_image_typing(image: ImageSource) -> bool:
    """Check if the given object respects typing of image.
    :param image: Image to check.
    :return: True if the object is an image, False otherwise.
    """
    if isinstance(image, get_args(SingleImageSource)):
        return True
    elif isinstance(image, list):
        return all([isinstance(image_item, get_args(SingleImageSource)) for image_item in image])
    else:
        return False


def is_image(filename: str) -> bool:
    """Check if the given file name refers to image.

    :param filename:    The filename to check.
    :return:            True if the file is an image, False otherwise.
    """
    return filename.split(".")[-1].lower() in IMG_EXTENSIONS
