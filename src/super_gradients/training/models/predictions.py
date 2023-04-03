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
        factory = BBoxFormatFactory()
        self.bboxes_xyxy = convert_bboxes(
            bboxes=bboxes,
            image_shape=image_shape,
            source_format=factory.get(bbox_format),
            target_format=factory.get("xyxy"),
            inplace=False,
        )
        self.confidence = confidence
        self.labels = labels
