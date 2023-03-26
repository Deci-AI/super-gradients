from typing import Union
import PIL

import numpy as np
import torch
import requests


def load_image(image: Union[str, np.ndarray, torch.Tensor, PIL.Image.Image]) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, torch.Tensor):
        return image.numpy()
    elif isinstance(image, PIL.Image.Image):
        return np.array(image.convert("RGB"))[:, :, ::-1].copy()
    elif isinstance(image, str):
        image = load_pil_image_from_str(image)
        return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def load_pil_image_from_str(image_str: str) -> PIL.Image.Image:
    if image_str.startswith("http://") or image_str.startswith("https://"):
        image = requests.get(image_str, stream=True).raw
        return PIL.Image.open(image)
    else:
        return PIL.Image.open(image_str)


def show_image(image: np.ndarray):
    PIL.Image.fromarray(image).show()


# images = [
#     np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[255, 0, 0], [255, 255, 0], [0, 0, 255]]]).astype(np.uint8),
#     torch.Tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[255, 0, 0], [255, 255, 0], [0, 0, 255]]]).to(dtype=torch.uint8),
#     "/Users/Louis.Dupont/Downloads/cat.jpeg",
#     "https://s.hs-data.com/bilder/spieler/gross/128069.jpg",
# ]
#
# for image in images:
#     show_image(load_image(image))
