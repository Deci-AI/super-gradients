from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass
from matplotlib import pyplot as plt

import numpy as np

from super_gradients.training.models.predictions import Prediction, DetectionPrediction
from super_gradients.training.utils.visualization.detection import draw_bbox
from super_gradients.training.utils.visualization.utils import generate_color_mapping


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

    def draw(
        self,
        box_thickness: int = 2,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int]]] = None,
    ) -> np.ndarray:
        """Draw the predicted bboxes on the image.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        :return:                Image with predicted bboxes. Note that this does not modify the original image.
        """
        image_np = self.image.copy()

        color_mapping = color_mapping or generate_color_mapping(len(self.class_names))

        for pred_i in range(len(self.predictions)):

            class_id = int(self.predictions.labels[pred_i])
            bbox_color = color_mapping[class_id]
            class_name = self.class_names[class_id]

            if show_confidence:
                class_name += f" {str(round(self.predictions.confidence[pred_i], 2))}"

            draw_bbox(
                image=image_np,
                title=class_name,
                color=bbox_color,
                box_thickness=box_thickness,
                x1=int(self.predictions.bboxes_xyxy[pred_i, 0]),
                y1=int(self.predictions.bboxes_xyxy[pred_i, 1]),
                x2=int(self.predictions.bboxes_xyxy[pred_i, 2]),
                y2=int(self.predictions.bboxes_xyxy[pred_i, 3]),
            )

        return image_np

    def show(
        self,
        box_thickness: int = 2,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int]]] = None,
    ) -> None:
        """Display the image with predicted bboxes.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        """
        image = self.draw(box_thickness=box_thickness, show_confidence=show_confidence, color_mapping=color_mapping)
        show_image(image=image)


def show_image(image: np.ndarray) -> None:
    """Plot an RGB image"""

    plt.figure(figsize=(image.shape[1] / 100.0, image.shape[0] / 100.0))
    plt.imshow(image, interpolation="nearest")
    plt.axis("off")
    plt.show()


@dataclass
class DetectionResults(Results):
    """Results of a detection task.

    :attr results:  List of the predictions results
    """

    def __init__(self, images: List[np.ndarray], predictions: List[DetectionPrediction], class_names: List[str]):
        self.results: List[DetectionResult] = []
        for image, prediction in zip(images, predictions):
            self.results.append(DetectionResult(image=image, predictions=prediction, class_names=class_names))

    def draw(
        self,
        box_thickness: int = 2,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int]]] = None,
    ) -> List[np.ndarray]:
        """Draw the predicted bboxes on the images.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        :return:                List of Images with predicted bboxes for each image. Note that this does not modify the original images.
        """
        return [prediction.draw(box_thickness=box_thickness, show_confidence=show_confidence, color_mapping=color_mapping) for prediction in self.results]

    def show(
        self,
        box_thickness: int = 2,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int]]] = None,
    ) -> None:
        """Display the predicted bboxes on the images.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        """
        for prediction in self.results:
            prediction.show(box_thickness=box_thickness, show_confidence=show_confidence, color_mapping=color_mapping)
