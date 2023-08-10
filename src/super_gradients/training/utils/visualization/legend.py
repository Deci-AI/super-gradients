from typing import Iterable, List, Sequence, Tuple
import cv2
import numpy as np
from dataclasses import dataclass

from super_gradients.training.utils.visualization.utils import best_text_color

FONT = cv2.FONT_HERSHEY_SIMPLEX
INITIAL_FONT_SIZE = 1
LINE_TYPE = max(int(INITIAL_FONT_SIZE * 2), 1)
MARGIN_SPACE = 20


@dataclass
class LabelInfo:
    """Hold information about labels.

    :attr name: Label name.
    :attr color: Color of the label.
    :attr text_size: Size of the label text.
    """

    name: str
    color: Tuple[int, int, int]
    text_size: Tuple[int, int]


@dataclass
class Row:
    """Represent a row of labels."""

    labels: List[LabelInfo]
    total_width: int


def get_text_size(text: str) -> Tuple[int, int]:
    """Calculate the size of a given text using the CV2 getTextSize function.

    :param text: Input text.
    :return: A tuple of width and height of the text box.
    """
    return cv2.getTextSize(text, FONT, INITIAL_FONT_SIZE, LINE_TYPE)[0]


def get_label_info(name: str, color: Tuple[int, int, int]) -> LabelInfo:
    """Creates a LabelInfo object for a given name and color.

    :param name: Label name.
    :param color: Label color.
    :return: An object of LabelInfo.
    """
    return LabelInfo(name, color, get_text_size(name))


def add_to_row_or_create_new(rows: List[Row], label: LabelInfo, image_width: int) -> List[Row]:
    """Adds a label to a row or creates a new row if the current one is full.

    :param rows: Existing rows of labels.
    :param label: Label to add.
    :param image_width: Width of the image.
    :return: Updated rows of labels.
    """
    if not rows or rows[-1].total_width + label.text_size[0] + 2 * MARGIN_SPACE > image_width:
        # create a new row and initialize total width
        rows.append(Row([label], label.text_size[0] + 2 * MARGIN_SPACE))
    else:
        # append label to existing row and add to total width
        rows[-1].labels.append(label)
        rows[-1].total_width += label.text_size[0] + MARGIN_SPACE
    return rows


def get_sorted_labels(class_color_tuples: Sequence[Tuple[str, Tuple[int, int, int]]]) -> List[LabelInfo]:
    """Sorts and creates LabelInfo for class-color tuples.

    :param class_color_tuples: Tuples of class names and associated colors.
    :return: A sorted list of LabelInfo objects.
    """
    sorted_classes = sorted(class_color_tuples, key=lambda x: x[0])
    return [get_label_info(name, color) for name, color in sorted_classes]


def get_label_rows(labels: List[LabelInfo], image_width: int) -> List[Row]:
    """Arranges labels in rows to fit into the image.

    :param labels: List of labels.
    :param image_width: Width of the image.
    :return: List of label rows.
    """
    rows = []
    for label in labels:
        rows = add_to_row_or_create_new(rows, label, image_width)
    return rows


def draw_label_on_canvas(canvas: np.ndarray, label: LabelInfo, position: Tuple[int, int], font_size: int) -> Tuple[np.ndarray, int]:
    """Draws a label on the canvas.

    :param canvas: The canvas to draw on.
    :param label: The label to draw.
    :param position: Position to draw the label.
    :param font_size: Font size of the label.
    :return: The updated canvas and horizontal position for next label.
    """
    upper_left = (position[0] - MARGIN_SPACE // 2, position[1] - label.text_size[1] - MARGIN_SPACE // 2)
    lower_right = (position[0] + label.text_size[0] + MARGIN_SPACE // 2, position[1] + MARGIN_SPACE // 2)
    canvas = cv2.rectangle(canvas, upper_left, lower_right, label.color, -1)
    canvas = cv2.putText(canvas, label.name, position, FONT, font_size, best_text_color(label.color), LINE_TYPE, lineType=cv2.LINE_AA)
    return canvas, position[0] + label.text_size[0] + MARGIN_SPACE


def draw_legend_on_canvas(image: np.ndarray, class_color_tuples: Iterable[Tuple[str, Tuple[int, int, int]]]) -> np.ndarray:
    """Draws a legend on the canvas.

    :param image: The image to draw the legend on.
    :param class_color_tuples: Iterable of tuples containing class name and its color.
    :return: The canvas with the legend drawnOops, it seems like the response got cut off.
    """
    sorted_labels = get_sorted_labels(class_color_tuples)
    label_rows = get_label_rows(sorted_labels, image.shape[1])

    canvas_height = (sorted_labels[0].text_size[1] + MARGIN_SPACE) * len(label_rows)
    canvas = np.ones((canvas_height, image.shape[1], 3), dtype=np.uint8) * 255

    vertical_position = sorted_labels[0].text_size[1] + MARGIN_SPACE // 2

    for row in label_rows:
        horizontal_position = MARGIN_SPACE
        for label in row.labels:
            canvas, horizontal_position = draw_label_on_canvas(canvas, label, (horizontal_position, vertical_position), INITIAL_FONT_SIZE)
        vertical_position += sorted_labels[0].text_size[1] + MARGIN_SPACE

    return canvas
