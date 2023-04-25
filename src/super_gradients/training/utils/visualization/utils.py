from typing import Tuple, List
import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_text_box(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    font: int,
    font_size: float,
    background_color: Tuple[int, int, int],
    thickness: int = 1,
) -> np.ndarray:
    """Draw a text inside a box

    :param image:               The image on which to draw the text box.
    :param text:                The text to display in the text box.
    :param x:                   The x-coordinate of the top-left corner of the text box.
    :param y:                   The y-coordinate of the top-left corner of the text box.
    :param font:                The font to use for the text.
    :param font_size:           The size of the font to use.
    :param background_color:    The color of the text box and text as a tuple of three integers representing RGB values.
    :param thickness:           The thickness of the text.
    :return: Image with the text inside the box.
    """
    text_color = best_text_color(background_color)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_size, thickness)
    text_left_offset = 7

    image = cv2.rectangle(image, (x, y), (x + text_width + text_left_offset, y - text_height - int(15 * font_size)), background_color, -1)
    image = cv2.putText(image, text, (x + text_left_offset, y - int(10 * font_size)), font, font_size, text_color, thickness, lineType=cv2.LINE_AA)
    return image


def best_text_color(background_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Determine the best color for text to be visible on a given background color.

    :param background_color: RGB values of the background color.
    :return: RGB values of the best text color for the given background color.
    """

    # If the brightness is greater than 0.5, use black text; otherwise, use white text.
    if compute_brightness(background_color) > 0.5:
        return (0, 0, 0)  # Black
    else:
        return (255, 255, 255)  # White


def compute_brightness(color: Tuple[int, int, int]) -> float:
    """Computes the brightness of a given color in RGB format. From https://alienryderflex.com/hsp.html

    :param color: A tuple of three integers representing the RGB values of the color.
    :return: The brightness of the color.
    """
    return (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[0]) / 255


def generate_color_mapping(num_classes: int) -> List[Tuple[int, ...]]:
    """Generate a unique BGR color for each class

    :param num_classes: The number of classes in the dataset.
    :return:            List of RGB colors for each class.
    """
    cmap = plt.cm.get_cmap("gist_rainbow", num_classes)
    colors = [cmap(i, bytes=True)[:3][::-1] for i in range(num_classes)]
    return [tuple(int(v) for v in c) for c in colors]
