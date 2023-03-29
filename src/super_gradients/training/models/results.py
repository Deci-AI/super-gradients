from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass
from matplotlib import pyplot as plt

import numpy as np

from super_gradients.training.utils.detection_utils import DetectionVisualization


@dataclass
class Result(ABC):
    image: np.ndarray
    class_names: List[str]

    @abstractmethod
    def show(self):
        pass


@dataclass
class Results(ABC):
    results: List[Result]

    @abstractmethod
    def show(self):
        pass


@dataclass
class DetectionResult(Result):
    image: np.ndarray
    _predictions: np.ndarray  # (N, 6) [X1, Y1, X2, Y2, score, class_id]
    class_names: List[str]

    @property
    def bboxes_xyxy(self) -> np.ndarray:
        return self._predictions[:, :4]

    @bboxes_xyxy.setter
    def bboxes_xyxy(self, value):
        self._predictions[:, :4] = value

    @property
    def confidence(self) -> np.ndarray:
        return self._predictions[:, 4]

    @property
    def class_ids(self) -> np.ndarray:
        return self._predictions[:, 5]

    def show(self, box_thickness: int = 2, show_confidence: bool = True, color_mapping: Optional[List[Tuple[int]]] = None):
        color_mapping = color_mapping or DetectionVisualization._generate_color_mapping(len(self.class_names))

        image_np = self.image.copy()

        for i in range(len(self._predictions)):
            image_np = DetectionVisualization._draw_box_title(
                color_mapping=color_mapping,
                class_names=self.class_names,
                box_thickness=box_thickness,
                image_np=image_np,
                x1=int(self.bboxes_xyxy[i, 0]),
                y1=int(self.bboxes_xyxy[i, 1]),
                x2=int(self.bboxes_xyxy[i, 2]),
                y2=int(self.bboxes_xyxy[i, 3]),
                class_id=int(self.class_ids[i]),
                pred_conf=self.confidence[i] if show_confidence else None,
            )

        plt.imshow(image_np, interpolation="nearest")
        plt.axis("off")
        plt.show()


@dataclass
class DetectionResults(Results):
    def __init__(self, images: List[np.ndarray], predictions: List[np.ndarray], class_names: List[str]):
        self.results: List[DetectionResult] = []
        for image, prediction in zip(images, predictions):
            self.results.append(DetectionResult(image=image, _predictions=prediction, class_names=class_names))

    def show(self, box_thickness: int = 2, show_confidence: bool = True, color_mapping: Optional[List[Tuple[int]]] = None):
        for prediction in self.results:
            prediction.show(box_thickness=box_thickness, color_mapping=color_mapping)
