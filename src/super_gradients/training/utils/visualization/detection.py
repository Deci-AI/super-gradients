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

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    diag_length = np.sqrt(bbox_width**2 + bbox_height**2)

    if box_thickness is None:
        # Calculate bbox thickness as a percentage of the geometric mean of bbox width and height
        box_thickness = int(max(1, diag_length * 0.007))

    # Draw bbox
    overlay = image.copy()
    overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)

    if title is not None or title != "":
        # Adapt font size to image shape.
        # This is required because small images require small font size, but this makes the title look bad,
        # so when possible we increase the font size to a more appropriate value.

        # Calculate base font size relative to bbox height
        base_font_scale_factor = 0.0025
        base_font_size = base_font_scale_factor * diag_length

        # Adjust font size based on image size (smaller images get smaller font size)
        image_size_factor = min(image.shape[:2]) / 1000  # Normalize by 1000 (or choose another suitable normalization factor)
        adjusted_font_size = base_font_size * image_size_factor

        # Apply minimum and maximum bounds
        min_font_size = 0.4
        max_font_size = 0.7
        font_size = max(min_font_size, adjusted_font_size)
        font_size = min(max_font_size, font_size)

        overlay = draw_text_box(image=overlay, text=title, x=x1, y=y1, font=2, font_size=font_size, background_color=color, thickness=1)

    return cv2.addWeighted(overlay, 0.75, image, 0.25, 0)
