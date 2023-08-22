from typing import Union, List, Tuple

import cv2
import numpy as np

from super_gradients.training.utils.visualization.detection import draw_bbox


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    score: float,
    edge_links: Union[np.ndarray, List[Tuple[int, int]]],
    edge_colors: Union[None, np.ndarray, List[Tuple[int, int, int]]],
    joint_thickness: int,
    keypoint_colors: Union[None, np.ndarray, List[Tuple[int, int, int]]],
    keypoint_radius: int,
    show_confidence: bool,
    box_thickness: int,
    keypoint_confidence_threshold: float = 0.0,
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
    :param keypoint_confidence_threshold: If keypoints contains confidence scores (Shape is [Num Joints, 3]), this function
    will draw keypoints with confidence score > threshold.


    :return: A new image with the skeleton drawn on it
    """
    image = image.copy()
    if edge_colors is None:
        edge_colors = [(255, 0, 255)] * len(edge_links)

    if keypoint_colors is None:
        keypoint_colors = [(0, 255, 0)] * len(keypoints)

    if len(edge_links) != len(edge_colors):
        raise ValueError("edge_colors and edge_links must have the same length")

    if len(keypoints.shape) != 2 or keypoints.shape[1] not in (2, 3):
        raise ValueError(f"Argument keypoints must be a 2D array of shape [Num Joints, 2] or [Num Joints, 3], got input of shape {keypoints.shape}")

    if keypoints.shape[1] == 3:
        keypoint_scores = keypoints[..., 2]
        keypoints = keypoints[..., 0:2].astype(int)
    else:
        # If keypoints contains no scores, set all scores above keypoint_confidence_threshold to draw them all
        keypoint_scores = np.ones(len(keypoints)) + keypoint_confidence_threshold
        keypoints = keypoints[..., 0:2].astype(int)

    keypoints_to_show_mask = keypoint_scores > keypoint_confidence_threshold

    for keypoint, show, color in zip(keypoints, keypoints_to_show_mask, keypoint_colors):
        if not show:
            continue
        x, y = keypoint
        x = int(x)
        y = int(y)
        color = tuple(map(int, color))
        cv2.circle(image, center=(x, y), radius=keypoint_radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    for (kp1, kp2), color in zip(edge_links, edge_colors):
        show = keypoints_to_show_mask[kp1] and keypoints_to_show_mask[kp2]
        if not show:
            continue
        p1 = tuple(map(int, keypoints[kp1]))
        p2 = tuple(map(int, keypoints[kp2]))
        color = tuple(map(int, color))
        cv2.line(image, p1, p2, color=color, thickness=joint_thickness, lineType=cv2.LINE_AA)

    confident_keypoints = keypoints[keypoints_to_show_mask]

    if show_confidence and len(confident_keypoints):
        x, y, w, h = cv2.boundingRect(confident_keypoints)
        image = draw_bbox(image, title=f"{score:.2f}", box_thickness=box_thickness, color=(255, 0, 255), x1=x, y1=y, x2=x + w, y2=y + h)

    return image
