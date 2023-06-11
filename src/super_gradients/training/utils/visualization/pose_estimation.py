from typing import Union, List, Tuple

import cv2
import numpy as np

from super_gradients.training.utils.visualization.detection import draw_bbox


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    score: float,
    edge_links: np.ndarray,
    edge_colors: Union[None, np.ndarray, List[Tuple[int, int, int]]],
    joint_thickness: int,
    keypoint_colors: Union[None, np.ndarray, List[Tuple[int, int, int]]],
    keypoint_radius: int,
    show_confidence: bool,
    box_thickness: int,
):
    """
    Draw a skeleton on an image.

    :param image: Input image (will not be modified)
    :param keypoints: Array of [Num Joints, 2] or [Num Joints, 3] containing the keypoints to draw.
                      First two values are the x and y coordinates, the third (optional, not used) is the confidence score.
    :param score:     Confidence score of the whole pose
    :param edge_links: Array of [Num Links, 2] containing the links between joints to draw.
    :param edge_colors: Array of shape [Num Links, 3] or list of tuples containing the (r,g,b) colors for each joint link.
    :param joint_thickness: Thickness of the joint links
    :param keypoint_colors: Array of shape [Num Joints, 3] or list of tuples containing the (r,g,b) colors for each keypoint.
    :param keypoint_radius: Radius of the keypoints (in pixels)
    :param show_confidence: Whether to show the bounding box around the pose and confidence score on top of it.
    :param box_thickness:   Thickness of bounding boxes.

    :return: A new image with the skeleton drawn on it
    """
    image = image.copy()
    if edge_colors is None:
        edge_colors = [(255, 0, 255)] * len(edge_links)

    if keypoint_colors is None:
        keypoint_colors = [(0, 255, 0)] * len(keypoints)

    if len(edge_links) != len(edge_colors):
        raise ValueError("edge_colors and edge_links must have the same length")

    keypoints = keypoints[..., 0:2].astype(int)

    for keypoint, color in zip(keypoints, keypoint_colors):
        color = tuple(map(int, color))
        cv2.circle(image, tuple(keypoint[:2]), radius=keypoint_radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    for joint, color in zip(edge_links, edge_colors):
        p1 = tuple(keypoints[joint[0]][:2])
        p2 = tuple(keypoints[joint[1]][:2])
        color = tuple(map(int, color))
        cv2.line(image, p1, p2, color=color, thickness=joint_thickness, lineType=cv2.LINE_AA)

    if show_confidence:
        x, y, w, h = cv2.boundingRect(keypoints)
        image = draw_bbox(image, title=f"{score:.2f}", box_thickness=box_thickness, color=(255, 0, 255), x1=x, y1=y, x2=x + w, y2=y + h)

    return image
