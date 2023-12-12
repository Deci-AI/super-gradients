from typing import Union, List, Tuple, Optional
import math

import cv2
import numpy as np

from super_gradients.training.utils.visualization.detection import draw_bbox, get_recommended_box_thickness


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    score: float,
    edge_links: Union[None, np.ndarray, List[Tuple[int, int]]],
    edge_colors: Union[None, np.ndarray, List[Tuple[int, int, int]]],
    joint_thickness: int,
    keypoint_colors: Union[None, np.ndarray, List[Tuple[int, int, int]]],
    keypoint_radius: int,
    show_confidence: bool,
    box_thickness: Optional[int],
    keypoint_confidence_threshold: float = 0.0,
    show_keypoint_confidence: bool = False,
):
    """
    Draw a skeleton on an image.

    :param image:           Input image (will not be modified)
    :param keypoints:       Array of [Num Joints, 2] or [Num Joints, 3] containing the keypoints to draw.
                            First two values are the x and y coordinates, the third (optional, not used) is the confidence score.
    :param score:           Confidence score of the whole pose
    :param edge_links:      Array of [Num Links, 2] containing the links between joints to draw. Can be None, in which case no links will be drawn.
    :param edge_colors:     Array of shape [Num Links, 3] or list of tuples containing the (r,g,b) colors for each joint link.
    :param joint_thickness: (Optional) Thickness of the joint links
    :param keypoint_colors: Array of shape [Num Joints, 3] or list of tuples containing the (r,g,b) colors for each keypoint.
    :param keypoint_radius: (Optional) Radius of the keypoints (in pixels)
    :param show_confidence: Whether to show the bounding box around the pose and confidence score on top of it.
    :param box_thickness:   (Optional) Thickness of bounding boxes. If None, will adapt to the box size.
    :param keypoint_confidence_threshold: If keypoints contains confidence scores (Shape is [Num Joints, 3]), this function
    will draw keypoints with confidence score > threshold.
    :param show_keypoint_confidence: Whether to show the confidence score for each keypoint individually.


    :return: A new image with the skeleton drawn on it
    """
    if edge_links is not None and edge_colors is not None and len(edge_links) != len(edge_colors):
        raise ValueError("edge_colors and edge_links must have the same length")

    if edge_colors is None and edge_links is not None:
        edge_colors = [(255, 0, 255)] * len(edge_links)

    if keypoint_colors is None:
        keypoint_colors = [(0, 255, 0)] * len(keypoints)

    if len(keypoints.shape) != 2 or keypoints.shape[1] not in (2, 3):
        raise ValueError(f"Argument keypoints must be a 2D array of shape [Num Joints, 2] or [Num Joints, 3], got input of shape {keypoints.shape}")

    if keypoints.shape[1] == 3:
        keypoint_scores = keypoints[..., 2]
        keypoints = keypoints[..., 0:2].astype(int)
    else:
        # If keypoints contains no scores, set all scores above keypoint_confidence_threshold to draw them all
        keypoint_scores = np.ones(len(keypoints)) + keypoint_confidence_threshold
        keypoints = keypoints[..., 0:2].astype(int)
        show_keypoint_confidence = False

    keypoints_to_show_mask = keypoint_scores > keypoint_confidence_threshold

    pose_center = keypoints[keypoints_to_show_mask].mean(axis=0)
    direction_from_center = keypoints - pose_center
    direction_from_center /= np.linalg.norm(direction_from_center, axis=1, ord=2, keepdims=True) + 1e-9

    overlay = image.copy()

    for keypoint, score, direction, show, color in zip(keypoints, keypoint_scores, direction_from_center, keypoints_to_show_mask, keypoint_colors):
        if not show:
            continue
        x, y = keypoint
        x = int(x)
        y = int(y)
        color = tuple(map(int, color))
        cv2.circle(overlay, center=(x, y), radius=keypoint_radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

        # Draw confidence score for each keypoint individually
        if show_keypoint_confidence:
            center_of_score = keypoint + direction * 16
            text = f"{score:.2f}"
            (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cx, cy = center_of_score
            if direction[0] < -0.5:
                x = int(cx - w)
            elif direction[0] > 0.5:
                x = int(cx)
            else:
                x = int(cx - w // 2)

            y = int(cy + h // 2)
            cv2.putText(overlay, text, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(250, 250, 250), thickness=1, lineType=cv2.LINE_AA)

    if edge_links is not None:
        for (kp1, kp2), color in zip(edge_links, edge_colors):
            show = keypoints_to_show_mask[kp1] and keypoints_to_show_mask[kp2]
            if not show:
                continue
            p1 = tuple(map(int, keypoints[kp1]))
            p2 = tuple(map(int, keypoints[kp2]))
            color = tuple(map(int, color))
            cv2.line(overlay, p1, p2, color=color, thickness=joint_thickness, lineType=cv2.LINE_AA)

    confident_keypoints = keypoints[keypoints_to_show_mask]

    if show_confidence and len(confident_keypoints):
        x, y, w, h = cv2.boundingRect(confident_keypoints)
        overlay = draw_bbox(overlay, title=f"{score:.2f}", box_thickness=box_thickness, color=(255, 0, 255), x1=x, y1=y, x2=x + w, y2=y + h)

    return cv2.addWeighted(overlay, 0.75, image, 0.25, 0)


class PoseVisualization:
    @classmethod
    def draw_poses(
        self,
        *,
        image: np.ndarray,
        poses: np.ndarray,
        boxes: Optional[np.ndarray],
        scores: Optional[np.ndarray],
        is_crowd: Optional[np.ndarray],
        edge_links: Union[np.ndarray, List[Tuple[int, int]]],
        edge_colors: Union[None, np.ndarray, List[Tuple[int, int, int]]],
        keypoint_colors: Union[None, np.ndarray, List[Tuple[int, int, int]]],
        show_keypoint_confidence: bool = False,
        joint_thickness: Optional[int] = None,
        box_thickness: Optional[int] = None,
        keypoint_radius: Optional[int] = None,
        keypoint_confidence_threshold: float = 0.5,
    ):
        """
        Draw multiple poses on an image.
        :param image: Image on which to draw the poses. This image will not be modified, instead a new image will be returned.
        :param poses: Predicted poses. Shape [Num Poses, Num Joints, 2] or [Num Poses, Num Joints, 3] if confidence scores are available.
        :param boxes: Optional bounding boxes for each pose. Shape [Num Poses, 4] in XYXY format.
        :param scores: Optional confidence scores for each pose. Shape [Num Poses]
        :param is_crowd: Optional array of booleans indicating whether each pose is crowd or not. Shape [Num Poses]
        :param edge_links: Array of [Num Links, 2] containing the links between joints to draw.
        :param edge_colors: Array of shape [Num Links, 3] or list of tuples containing the (r,g,b) colors for each joint link.
        :param keypoint_colors: Array of shape [Num Joints, 3] or list of tuples containing the (r,g,b) colors for each keypoint.
        :param show_keypoint_confidence: Whether to show the confidence score for each keypoint individually.
        :param keypoint_confidence_threshold: A minimal confidence score for individual keypoint to be drawn.
        :param joint_thickness: (Optional) Thickness of the joint links
        :return: A new image with the poses drawn on it.
        """
        if boxes is not None and len(boxes) != len(poses):
            raise ValueError("boxes and poses must have the same length")
        if scores is not None and len(scores) != len(poses):
            raise ValueError("conf and poses must have the same length")
        if is_crowd is not None and len(is_crowd) != len(poses):
            raise ValueError("is_crowd and poses must have the same length")

        # For visualization purposes, sort poses by confidence starting from the least confident
        if scores is not None:
            order = np.argsort(scores)
            poses = poses[order]
            scores = scores[order]
            if boxes is not None:
                boxes = boxes[order]
            if is_crowd is not None:
                is_crowd = is_crowd[order]

        res_image = image.copy()
        num_poses = len(poses)

        for pose_index in range(num_poses):

            if boxes is not None:
                x1 = int(boxes[pose_index][0])
                y1 = int(boxes[pose_index][1])
                x2 = int(boxes[pose_index][2])
                y2 = int(boxes[pose_index][3])

                current_box_thickness = box_thickness or get_recommended_box_thickness(x1, y1, x2, y2)
                current_joint_thickness = joint_thickness or current_box_thickness
                current_keypoint_radius = keypoint_radius or math.ceil(current_box_thickness * 3 / 2)
            else:
                current_joint_thickness = 2
                current_keypoint_radius = 3
                current_box_thickness = 2

            res_image = draw_skeleton(
                image=res_image,
                keypoints=poses[pose_index],
                score=scores[pose_index] if scores is not None else None,
                edge_links=edge_links,
                edge_colors=edge_colors,
                joint_thickness=current_joint_thickness,
                keypoint_colors=keypoint_colors,
                keypoint_radius=current_keypoint_radius,
                show_confidence=scores is not None and boxes is None,
                show_keypoint_confidence=show_keypoint_confidence,
                box_thickness=current_box_thickness,
                keypoint_confidence_threshold=keypoint_confidence_threshold,
            )

            if boxes is not None:
                x1 = int(boxes[pose_index][0])
                y1 = int(boxes[pose_index][1])
                x2 = int(boxes[pose_index][2])
                y2 = int(boxes[pose_index][3])
                title = ""
                if scores is not None:
                    title += f"{scores[pose_index]:.2f}"
                if is_crowd is not None:
                    title += f"Crowd {is_crowd[pose_index]}"

                res_image = draw_bbox(
                    image=res_image,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    color=(255, 255, 255),
                    title=title,
                    box_thickness=current_box_thickness,
                )

        return res_image
