from dataclasses import dataclass

import numpy as np

from super_gradients.training.utils.detection_utils import DetectionVisualization
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST


@dataclass
class Prediction:
    _image: np.ndarray
    _boxes: np.ndarray  # (N, 4)
    _classes: np.ndarray  # (N,)
    _scores: np.ndarray  # (N,)
    _image: np.ndarray  # (H, W, 3)

    def show(self, class_colors=None):

        box_thickness: int = 2
        image_scale: float = 1.0

        class_names = COCO_DETECTION_CLASSES_LIST

        image_np = self._image[:, :, ::-1].copy()
        color_mapping = DetectionVisualization._generate_color_mapping(len(class_names))

        # Draw predictions
        self._boxes *= image_scale
        for box in self._boxes:
            image_np = DetectionVisualization._draw_box_title(
                color_mapping=color_mapping,
                class_names=class_names,
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
