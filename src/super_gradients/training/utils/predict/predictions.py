from typing import Tuple, Optional
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

        :param target_bboxes: np.ndarray, ground truth bounding boxes as np.ndarray of shape (image_i_object_count, 4)
         When not None, will plot the predictions and the ground truth bounding boxes side by side (i.e 2 images stitched as one)

        :param target_labels: np.ndarray, ground truth target class indices as an np.ndarray of shape (image_i_object_count).

        :param target_bbox_format: str, bounding box format of target_bboxes, one of ['xyxy','xywh',
        'yxyx' 'cxcywh' 'normalized_xyxy' 'normalized_xywh', 'normalized_yxyx', 'normalized_cxcywh']. Will raise an
        error if not None and target_bboxes is None.
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
        self.image_shape = image_shape

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

    :param poses:  Numpy array of [Num Poses, Num Joints, 2] shape
    :param scores: Numpy array of [Num Poses] shape
    :param boxes:  Numpy array of [Num Poses, 4] shape which represents the bounding boxes of each pose in xyxy format
    """

    poses: np.ndarray
    scores: np.ndarray
    bboxes_xyxy: Optional[np.ndarray]
    edge_links: np.ndarray
    edge_colors: np.ndarray
    keypoint_colors: np.ndarray
    image_shape: Tuple[int, int]

    def __init__(
        self,
        poses: np.ndarray,
        scores: np.ndarray,
        bboxes_xyxy: Optional[np.ndarray],
        edge_links: np.ndarray,
        edge_colors: np.ndarray,
        keypoint_colors: np.ndarray,
        image_shape: Tuple[int, int],
    ):
        """
        :param poses:       Predicted poses as a numpy array of shape [Num Poses, Num Joints, 2]
        :param scores:      Confidence scores for each pose [Num Poses]
        :param bboxes_xyxy:      Bounding boxes of each pose in xyxy format [Num Poses, 4]
        :param image_shape: Shape of the image the prediction is made on, (H, W).
        """
        self._validate_input(poses, scores, bboxes_xyxy, edge_links, edge_colors, keypoint_colors)
        self.poses = poses
        self.scores = scores
        self.bboxes_xyxy = bboxes_xyxy
        self.edge_links = edge_links
        self.edge_colors = edge_colors
        self.image_shape = image_shape
        self.keypoint_colors = keypoint_colors

    def _validate_input(self, poses: np.ndarray, scores: np.ndarray, bboxes: Optional[np.ndarray], edge_links, edge_colors, keypoint_colors) -> None:
        if not isinstance(poses, np.ndarray):
            raise ValueError(f"Argument poses must be a numpy array, not {type(poses)}")
        if not isinstance(scores, np.ndarray):
            raise ValueError(f"Argument scores must be a numpy array, not {type(scores)}")
        if bboxes is not None and not isinstance(bboxes, np.ndarray):
            raise ValueError(f"Argument bboxes must be a numpy array, not {type(bboxes)}")
        if not isinstance(keypoint_colors, np.ndarray):
            raise ValueError(f"Argument keypoint_colors must be a numpy array, not {type(keypoint_colors)}")
        if len(poses) != len(scores) != len(keypoint_colors):
            raise ValueError(f"The number of poses ({len(poses)}) does not match the number of scores ({len(scores)}).")
        if len(edge_links) != len(edge_colors):
            raise ValueError(f"The number of joint links ({len(edge_links)}) does not match the number of joint colors ({len(edge_colors)}).")

    def __len__(self):
        return len(self.poses)


@dataclass
class ClassificationPrediction(Prediction):
    """Represents a Classification prediction"""

    confidence: float
    label: int
    image_shape: Tuple[int, int]

    def __init__(self, confidence: float, label: int, image_shape: Optional[Tuple[int, int]]):
        """

        :param confidence:  Confidence scores for each bounding box
        :param label:      Labels for each bounding box.
        :param image_shape: Shape of the image the prediction is made on, (H, W).
        """
        self._validate_input(confidence, label)

        self.confidence = confidence
        self.label = label
        self.image_shape = image_shape

    def _validate_input(self, confidence: float, label: int) -> None:
        if not isinstance(confidence, float):
            raise ValueError(f"Argument confidence must be a float, not {type(confidence)}")
        if not isinstance(label, int):
            raise ValueError(f"Argument labels must be an integer, not {type(label)}")

    def __len__(self):
        return len(self.labels)


@dataclass
class SegmentationPrediction(Prediction):
    """Represents a segmentation prediction."""

    segmentation_map: np.ndarray
    segmentation_map_shape: Tuple[int, int]
    image_shape: Tuple[int, int]

    def __init__(self, segmentation_map: np.ndarray, segmentation_map_shape: Tuple[int, int], image_shape: Tuple[int, int]):
        """
        :param segmentation_map: Segmentation map (predication) in the shape specified segmentation_map_shape
        :param segmentation_map_shape: Shape of the prediction (H, W).
        :param image_shape: Shape of the image the prediction is made on, (H, W).
        """
        self._validate_input(segmentation_map_shape, image_shape)

        self.segmentation_map = segmentation_map
        self.segmentation_map_shape = segmentation_map_shape
        self.image_shape = image_shape

    def _validate_input(self, segmentation_map_shape: Tuple[int, int], image_shape: Tuple[int, int]) -> None:
        if segmentation_map_shape[0] != image_shape[0] or segmentation_map_shape[1] != image_shape[1]:
            raise ValueError("The shape of the segmentation map does not match the shape of the input image.")
