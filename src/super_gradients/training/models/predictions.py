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

    _bboxes: np.ndarray
    _bbox_format: str

    confidence: np.ndarray
    labels: np.ndarray
    image_shape: Tuple[int, int]

    def __init__(self, bboxes: np.ndarray, bbox_format: str, confidence: np.ndarray, labels: np.ndarray, image_shape: Tuple[int, int]):
        """
        :param bboxes:      BBoxes in the format specified by bbox_format
        :param bbox_format: BBoxes format that can be a string ("xyxy", "cxywh", ...)
        :param confidence:  Confidence scores for each bounding box
        :param labels:      Labels for each bounding box
        :param image_shape: Shape of the image the prediction is made on
        """
        self._bboxes = bboxes
        self._bbox_format = bbox_format
        self.confidence = confidence
        self.labels = labels
        self.image_shape = image_shape

    @property
    def bboxes_xyxy(self):
        return self._get_bbox_as("xyxy")

    @bboxes_xyxy.setter
    def bboxes_xyxy(self, bboxes: np.ndarray):
        self._set_bbox_from(bboxes=bboxes, input_bbox_format="xyxy")

    @property
    def bboxes_cxcywh(self):
        return self._get_bbox_as("cxcywh")

    @bboxes_cxcywh.setter
    def bboxes_cxcywh(self, bboxes: np.ndarray):
        self._set_bbox_from(bboxes=bboxes, input_bbox_format="cxcywh")

    def _get_bbox_as(self, desired_bbox_format: str):
        factory = BBoxFormatFactory()
        return convert_bboxes(
            bboxes=self._bboxes,
            image_shape=self.image_shape,
            source_format=factory.get(self._bbox_format),
            target_format=factory.get(desired_bbox_format),
            inplace=False,
        )

    def _set_bbox_from(self, bboxes: np.ndarray, input_bbox_format: str):
        factory = BBoxFormatFactory()
        self._bboxes = convert_bboxes(
            bboxes=bboxes,
            image_shape=self.image_shape,
            source_format=factory.get(input_bbox_format),
            target_format=factory.get(self._bbox_format),
            inplace=False,
        )
        self._bbox_format = input_bbox_format
