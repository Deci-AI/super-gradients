from typing import Union, List
import PIL

import numpy as np
import torch
import requests


ImageType = Union[str, np.ndarray, torch.Tensor, PIL.Image.Image]


def load_images(images: Union[List[ImageType], ImageType]) -> List[np.ndarray]:
    if isinstance(images, list):
        return [load_image(image=image) for image in images]
    else:
        return [load_image(image=images)]


def load_image(image: ImageType) -> np.ndarray:
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
    return np.asarray(image.convert("RGB"))  # TODO: Check RGB/BGR


def load_pil_image_from_str(image_str: str) -> PIL.Image.Image:
    """Load an image based on a string"""
    if image_str.startswith("http://") or image_str.startswith("https://"):
        image = requests.get(image_str, stream=True).raw
        return PIL.Image.open(image)
    else:
        return PIL.Image.open(image_str)


def show_image(image: np.ndarray):
    PIL.Image.fromarray(image).show()
