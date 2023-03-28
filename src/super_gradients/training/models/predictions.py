from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch

from super_gradients.training.utils.detection_utils import DetectionVisualization


@dataclass
class Prediction(ABC):
    image: np.ndarray
    class_names: List[str]

    @abstractmethod
    def show(self, class_colors=None):
        pass


@dataclass
class ClassificationPrediction(Prediction):
    image: np.ndarray
    _class: int
    class_names: List[str]

    def show(self, class_colors=None):
        raise NotImplementedError()


@dataclass
class SegmentationPrediction(Prediction):
    image: np.ndarray
    _mask: np.ndarray
    class_names: List[str]

    def show(self, class_colors=None):

        from torchvision.utils import draw_segmentation_masks

        bool_mask = np.zeros((self._mask.max(), *self._mask.shape), dtype=np.bool)
        for i in range(bool_mask.shape[0]):
            bool_mask[i, :, :] = self._mask == i

        image_np = self.image.copy()
        image_np = np.ascontiguousarray(image_np.transpose(2, 0, 1))
        image = draw_segmentation_masks(
            image=torch.from_numpy(image_np.astype(np.uint8)),
            masks=torch.from_numpy(bool_mask),
        )
        image = image.detach().cpu().numpy().astype(np.uint8)

        inverse_permutation = np.argsort(np.array((2, 0, 1)))
        image = np.ascontiguousarray(image.transpose(inverse_permutation))

        from matplotlib import pyplot as plt

        plt.imshow(image, interpolation="nearest")
        plt.show()


@dataclass
class DetectionPrediction(Prediction):
    image: np.ndarray
    _boxes: np.ndarray  # (N, 4)
    _classes: np.ndarray  # (N,)
    _scores: np.ndarray  # (N,)
    class_names: List[str]

    def show(self, class_colors=None):

        box_thickness: int = 2
        image_scale: float = 1.0

        image_np = self.image[:, :, ::-1].copy()
        color_mapping = DetectionVisualization._generate_color_mapping(len(self.class_names))

        # Draw predictions
        self._boxes *= image_scale
        for box in self._boxes:
            image_np = DetectionVisualization._draw_box_title(
                color_mapping=color_mapping,
                class_names=self.class_names,
                box_thickness=box_thickness,
                image_np=image_np,
                x1=int(box[0]),
                y1=int(box[1]),
                x2=int(box[2]),
                y2=int(box[3]),
                class_id=int(box[5]),
                pred_conf=box[4],
            )
        from matplotlib import pyplot as plt

        plt.imshow(image_np, interpolation="nearest")
        plt.show()
