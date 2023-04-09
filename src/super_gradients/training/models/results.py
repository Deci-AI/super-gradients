from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass
from matplotlib import pyplot as plt

import numpy as np

from super_gradients.training.utils.detection_utils import DetectionVisualization
from super_gradients.training.models.predictions import Prediction, DetectionPrediction


@dataclass
class Result(ABC):
    """Results of a given computer vision task (detection, classification, etc.).

    :attr image:        Input image
    :attr predictions:  Predictions of the model
    :attr class_names:  List of the class names to predict
    """

    image: np.ndarray
    predictions: Prediction
    class_names: List[str]

    @abstractmethod
    def draw(self) -> np.ndarray:
        """Draw the predictions on the image."""
        pass

    @abstractmethod
    def show(self) -> None:
        """Display the predictions on the image."""
        pass


@dataclass
class Results(ABC):
    """List of results of a given computer vision task (detection, classification, etc.).

    :attr results: List of results of the run
    """

    results: List[Result]

    @abstractmethod
    def draw(self) -> List[np.ndarray]:
        """Draw the predictions on the image."""
        pass

    @abstractmethod
    def show(self) -> None:
        """Display the predictions on the image."""
        pass


@dataclass
class DetectionResult(Result):
    """Result of a detection task.

    :attr image:        Input image
    :attr predictions:  Predictions of the model
    :attr class_names:  List of the class names to predict
    """

    image: np.ndarray
    predictions: DetectionPrediction
    class_names: List[str]

    def draw(self, box_thickness: int = 2, show_confidence: bool = True, color_mapping: Optional[List[Tuple[int]]] = None) -> np.ndarray:
        """Draw the predicted bboxes on the image.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        :return:                Image with predicted bboxes. Note that this does not modify the original image.
        """
        image_np = self.image.copy()
        color_mapping = color_mapping or DetectionVisualization._generate_color_mapping(len(self.class_names))

        for pred_i in range(len(self.predictions)):
            image_np = DetectionVisualization._draw_box_title(
                color_mapping=color_mapping,
                class_names=self.class_names,
                box_thickness=box_thickness,
                image_np=image_np,
                x1=int(self.predictions.bboxes_xyxy[pred_i, 0]),
                y1=int(self.predictions.bboxes_xyxy[pred_i, 1]),
                x2=int(self.predictions.bboxes_xyxy[pred_i, 2]),
                y2=int(self.predictions.bboxes_xyxy[pred_i, 3]),
                class_id=int(self.predictions.labels[pred_i]),
                pred_conf=self.predictions.confidence[pred_i] if show_confidence else None,
            )
        return image_np

    def show(self, box_thickness: int = 2, show_confidence: bool = True, color_mapping: Optional[List[Tuple[int]]] = None) -> None:
        """Display the image with predicted bboxes.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        """
        image_np = self.draw(box_thickness=box_thickness, show_confidence=show_confidence, color_mapping=color_mapping)

        plt.imshow(image_np, interpolation="nearest")
        plt.axis("off")
        plt.show()


@dataclass
class DetectionResults(Results):
    """Results of a detection task.

    :attr results:  List of the predictions results
    """

    results: List[DetectionResult]

    def draw(self, box_thickness: int = 2, show_confidence: bool = True, color_mapping: Optional[List[Tuple[int]]] = None) -> List[np.ndarray]:
        """Draw the predicted bboxes on the images.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        :return:                List of Images with predicted bboxes for each image. Note that this does not modify the original images.
        """
        return [prediction.draw(box_thickness=box_thickness, show_confidence=show_confidence, color_mapping=color_mapping) for prediction in self.results]

    def show(self, box_thickness: int = 2, show_confidence: bool = True, color_mapping: Optional[List[Tuple[int]]] = None) -> None:
        """Display the predicted bboxes on the images.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        """
        for prediction in self.results:
            prediction.show(box_thickness=box_thickness, show_confidence=show_confidence, color_mapping=color_mapping)
