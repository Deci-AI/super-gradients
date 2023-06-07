from typing import Tuple
from abc import ABC
from dataclasses import dataclass

import numpy as np

from super_gradients.common.factories.bbox_format_factory import BBoxFormatFactory
from super_gradients.training.datasets.data_formats.bbox_formats import convert_bboxes


@dataclass
class Prediction(ABC):
    pass


@dataclass
class DetectionPrediction(Prediction):
    """Represents a detection prediction, with bboxes represented in xyxy format."""

    bboxes_xyxy: np.ndarray
    confidence: np.ndarray
    labels: np.ndarray

    def __init__(self, bboxes: np.ndarray, bbox_format: str, confidence: np.ndarray, labels: np.ndarray, image_shape: Tuple[int, int]):
        """
        :param bboxes:      BBoxes in the format specified by bbox_format
        :param bbox_format: BBoxes format that can be a string ("xyxy", "cxywh", ...)
        :param confidence:  Confidence scores for each bounding box
        :param labels:      Labels for each bounding box.
        :param image_shape: Shape of the image the prediction is made on, (H, W). This is used to convert bboxes to xyxy format
        """
        self._validate_input(bboxes, confidence, labels)

        factory = BBoxFormatFactory()
        bboxes_xyxy = convert_bboxes(
            bboxes=bboxes,
            image_shape=image_shape,
            source_format=factory.get(bbox_format),
            target_format=factory.get("xyxy"),
            inplace=False,
        )

        self.bboxes_xyxy = bboxes_xyxy
        self.confidence = confidence
        self.labels = labels

    def _validate_input(self, bboxes: np.ndarray, confidence: np.ndarray, labels: np.ndarray) -> None:
        n_bboxes, n_confidences, n_labels = bboxes.shape[0], confidence.shape[0], labels.shape[0]
        if n_bboxes != n_confidences != n_labels:
            raise ValueError(
                f"The number of bounding boxes ({n_bboxes}) does not match the number of confidence scores ({n_confidences}) and labels ({n_labels})."
            )

    def __len__(self):
        return len(self.bboxes_xyxy)


@dataclass
class PoseEstimationPrediction(Prediction):
    """Represents a pose estimation prediction.

    :attr poses: Numpy array of [Num Poses, Num Joints, 2] shape
    :attr scores: Numpy array of [Num Poses] shape
    """

    poses: np.ndarray
    scores: np.ndarray
    edge_links: np.ndarray
    edge_colors: np.ndarray
    keypoint_colors: np.ndarray
    image_shape: Tuple[int, int]

    def __init__(
        self,
        poses: np.ndarray,
        scores: np.ndarray,
        edge_links: np.ndarray,
        edge_colors: np.ndarray,
        keypoint_colors: np.ndarray,
        image_shape: Tuple[int, int],
    ):
        """
        :param poses:
        :param scores:
        :param image_shape: Shape of the image the prediction is made on, (H, W). This is used to convert bboxes to xyxy format
        """
        self._validate_input(poses, scores, edge_links, edge_colors, keypoint_colors)
        self.poses = poses
        self.scores = scores
        self.edge_links = edge_links
        self.edge_colors = edge_colors
        self.image_shape = image_shape
        self.keypoint_colors = keypoint_colors

    def _validate_input(self, poses: np.ndarray, scores: np.ndarray, edge_links, edge_colors, keypoint_colors) -> None:
        if not isinstance(poses, np.ndarray):
            raise ValueError(f"Argument poses must be a numpy array, not {type(poses)}")
        if not isinstance(scores, np.ndarray):
            raise ValueError(f"Argument scores must be a numpy array, not {type(scores)}")
        if not isinstance(keypoint_colors, np.ndarray):
            raise ValueError(f"Argument keypoint_colors must be a numpy array, not {type(keypoint_colors)}")
        if len(poses) != len(scores) != len(keypoint_colors):
            raise ValueError(f"The number of poses ({len(poses)}) does not match the number of scores ({len(scores)}).")
        if len(edge_links) != len(edge_colors):
            raise ValueError(f"The number of joint links ({len(edge_links)}) does not match the number of joint colors ({len(edge_colors)}).")

    def __len__(self):
        return len(self.poses)
