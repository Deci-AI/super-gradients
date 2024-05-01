import random
from typing import Tuple, List, Dict

import numpy as np
from super_gradients.common.registry import register_transform
from .obb_sample import OBBSample

from .abstract_obb_transform import AbstractOBBDetectionTransform


@register_transform()
class OBBDetectionRandomRotate90(AbstractOBBDetectionTransform):
    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def apply_to_sample(self, sample: OBBSample) -> OBBSample:
        if random.random() < self.prob:
            k = random.randrange(0, 4)
            image_shape = sample.image.shape[:2]
            sample = OBBSample(
                image=self.apply_to_image(sample.image, k),
                bboxes_xyxy=self.apply_to_bboxes(sample.bboxes_xyxy, k, image_shape),
                labels=sample.labels,
                is_crowd=sample.is_crowd,
                additional_samples=None,
            )
        return sample

    def apply_to_image(self, image: np.ndarray, factor: int) -> np.ndarray:
        """
        Apply a `factor` number of 90-degree rotation to image.

        :param image:  Input image (HWC).
        :param factor: Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
        :return:       Rotated image (HWC).
        """
        return np.ascontiguousarray(np.rot90(image, factor))

    def apply_to_bboxes(self, bboxes: np.ndarray, factor: int, image_shape: Tuple[int, int]):
        """
        Apply a `factor` number of 90-degree rotation to bounding boxes.

        :param bboxes:       Input bounding boxes in XYXY format.
        :param factor:       Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
        :param image_shape:  Original image shape
        :return:             Rotated bounding boxes in XYXY format.
        """
        rows, cols = image_shape
        bboxes_rotated = self.xyxy_bbox_rot90(bboxes, factor, rows, cols)
        return bboxes_rotated

    @classmethod
    def xyxy_bbox_rot90(cls, bboxes: np.ndarray, factor: int, rows: int, cols: int):
        """
        Rotates a bounding box by 90 degrees CCW (see np.rot90)

        :param bboxes:  Tensor made of bounding box tuples (x_min, y_min, x_max, y_max).
        :param factor:  Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
        :param rows:    Image rows of the original image.
        :param cols:    Image cols of the original image.

        :return: A bounding box tuple (x_min, y_min, x_max, y_max).

        """
        x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        if factor == 0:
            bbox = x_min, y_min, x_max, y_max
        elif factor == 1:
            bbox = y_min, cols - x_max, y_max, cols - x_min
        elif factor == 2:
            bbox = cols - x_max, rows - y_max, cols - x_min, rows - y_min
        elif factor == 3:
            bbox = rows - y_max, x_min, rows - y_min, x_max
        else:
            raise ValueError("Parameter n must be in set {0, 1, 2, 3}")
        return np.stack(bbox, axis=1)

    def get_equivalent_preprocessing(self) -> List[Dict]:
        raise NotImplementedError("get_equivalent_preprocessing is not implemented for non-deterministic transforms.")
