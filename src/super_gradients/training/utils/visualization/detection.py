from typing import Tuple, Optional
import cv2
import numpy as np

from super_gradients.training.utils.visualization.utils import draw_text_box


def draw_bbox(
    image: np.ndarray,
    title: Optional[str],
    color: Tuple[int, int, int],
    box_thickness: Optional[int],
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

    if box_thickness is None:
        box_thickness = get_recommended_box_thickness(x1=x1, y1=y1, x2=x2, y2=y2)

    # Draw bbox
    overlay = image.copy()
    overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)

    if title is not None or title != "":
        # Adapt font size to image shape.
        # This is required because small images require small font size, but this makes the title look bad,
        # so when possible we increase the font size to a more appropriate value.

        font_size = get_recommended_text_size(x1=x1, y1=y1, x2=x2, y2=y2)
        overlay = draw_text_box(image=overlay, text=title, x=x1, y=y1, font=2, font_size=font_size, background_color=color, thickness=1)

    return cv2.addWeighted(overlay, 0.75, image, 0.25, 0)


def get_recommended_box_thickness(x1: int, y1: int, x2: int, y2: int) -> int:
    """Get a nice box thickness for a given bounding box."""
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    diag_length = np.sqrt(bbox_width**2 + bbox_height**2)

    if diag_length <= 100:
        return 1
    elif diag_length <= 200:
        return 2
    else:
        return 3


def get_recommended_text_size(x1: int, y1: int, x2: int, y2: int) -> float:
    """Get a nice text size for a given bounding box."""
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    diag_length = np.sqrt(bbox_width**2 + bbox_height**2)

    # This follows the heuristic (defined after some visual experiments):
    # - diag_length=100 -> base_font_size=0.4 (min text size)
    # - diag_length=300 -> base_font_size=0.7 (max text size)
    font_size = diag_length * 0.0015 + 0.25
    font_size = max(0.4, font_size)  # Min = 0.4
    font_size = min(0.7, font_size)  # Max = 0.7

    return font_size
