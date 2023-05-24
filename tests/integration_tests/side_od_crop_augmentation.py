import random
from typing import Optional
import numpy as np


class DetectionRandomSideCrop:
    """Crop a random part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, rel_x: float = 0.5, side: Optional[str] = None, p: float = 1.0):
        if side is not None:
            assert side in ["left", "right"], "side must be left or right"

        assert rel_x >= 0 and rel_x <= 0.5, "rel_min_x must be between 0 and 0.5"

        self.rel_x = rel_x
        self.side = side
        self.p = p

    def apply(self, img: np.ndarray, bboxes: np.ndarray, **params):
        if random.random() > self.p:
            return img, bboxes

        if self.side is None:
            self.side = random.choice(["left", "right"])

        random_rel_x = random.uniform(0, self.rel_x)
        abs_x = int(random_rel_x * img.shape[0])

        if self.side == "left":
            img = img[abs_x:]
            bboxes = self._crop_left_bboxes(bboxes, abs_x)

        elif self.side == "right":
            img = img[:abs_x]
            bboxes = self._crop_right_bboxes(bboxes, abs_x)

        return img, bboxes

    def _crop_right_bboxes(self, bboxes: np.ndarray, abs_x: int):
        fixed_bboxes = []

        for bbox in bboxes:
            if bbox[0] < abs_x:
                fixed_bboxes.append([bbox[0], bbox[1], min(bbox[2], abs_x), bbox[3]])

        return np.array(fixed_bboxes)

    def _crop_left_bboxes(self, bboxes: np.ndarray, abs_x: int):
        fixed_bboxes = []

        for bbox in bboxes:
            if bbox[2] > abs_x:
                fixed_bboxes.append([max(bbox[0], abs_x), bbox[1], bbox[2], bbox[3]])

        return np.array(fixed_bboxes)


if __name__ == "__main__":
    pass
