from typing import Sequence
import cv2
import numpy as np

from super_gradients.training.utils.visualization.utils import draw_text_box


def draw_bbox(
    image: np.ndarray,
    title: str,
    color: Sequence[int],
    box_thickness: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """Draw a bounding box on an image.

    :param image:           Image on which to draw the bounding box.
    :param color:           RGB values of the color of the bounding box.
    :param title:           Title to display inside the bounding box.
    :param box_thickness:   Thickness of the bounding box border.
    :param x1:              x-coordinate of the top-left corner of the bounding box.
    :param y1:              y-coordinate of the top-left corner of the bounding box.
    :param x2:              x-coordinate of the bottom-right corner of the bounding box.
    :param y2:              y-coordinate of the bottom-right corner of the bounding box.
    """

    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

    # Adapt font size to image shape.
    # This is required because small images require small font size, but this makes the title look bad,
    # so when possible we increase the font size to a more appropriate value.
    font_size = 0.25 + 0.07 * min(image.shape[:2]) / 100
    font_size = max(font_size, 0.5)  # Set min font_size to 0.5
    font_size = min(font_size, 0.8)  # Set max font_size to 0.8

    image = draw_text_box(image=image, text=title, x=x1, y=y1, font=2, font_size=font_size, background_color=color, thickness=1)

    return image
