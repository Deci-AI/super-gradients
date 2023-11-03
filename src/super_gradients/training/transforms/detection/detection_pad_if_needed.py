from typing import List

import numpy as np

from super_gradients.common.object_names import Transforms, Processings
from super_gradients.common.registry.registry import register_transform
from super_gradients.training.samples import DetectionSample
from super_gradients.training.transforms.utils import _pad_image, PaddingCoordinates, _shift_bboxes_xywh
from . import AbstractDetectionTransform


@register_transform(Transforms.DetectionPadIfNeeded)
class DetectionPadIfNeeded(AbstractDetectionTransform):
    """
    Pad image and targets to ensure that resulting image size is not less than (min_width, min_height).
    """

    def __init__(self, min_height: int, min_width: int, pad_value: int, padding_mode: str = "bottom_right"):
        """

        :param output_size:     Desired image size (rows, cols)
        :param pad_value: Padding value of image
        :param mask_pad_value:  Padding value for mask
        """
        if padding_mode not in ("bottom_right", "center"):
            raise ValueError(f"Unknown padding mode: {padding_mode}. Supported modes: 'bottom_right', 'center'")
        super().__init__()
        self.min_height = min_height
        self.min_width = min_width
        self.image_pad_value = pad_value
        self.padding_mode = padding_mode

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        height, width = sample.image.shape[:2]

        if self.padding_mode == "bottom_right":
            pad_left = 0
            pad_top = 0
            pad_bottom = max(0, self.min_height - height)
            pad_right = max(0, self.min_width - width)
        elif self.padding_mode == "center":
            pad_left = max(0, (self.min_width - width) // 2)
            pad_top = max(0, (self.min_height - height) // 2)
            pad_bottom = max(0, self.min_height - height - pad_top)
            pad_right = max(0, self.min_width - width - pad_left)
        else:
            raise RuntimeError(f"Unknown padding mode: {self.padding_mode}")

        padding_coordinates = PaddingCoordinates(top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right)

        sample.image = _pad_image(sample.image, padding_coordinates, self.image_pad_value)
        sample.bboxes_xywh = self.apply_to_bboxes(sample.bboxes_xywh, pad_left, pad_top)
        return sample

    def apply_to_bboxes(self, bboxes: np.ndarray, pad_left, pad_top):
        bboxes = _shift_bboxes_xywh(bboxes, shift_w=pad_left, shift_h=pad_top)
        return bboxes

    def get_equivalent_preprocessing(self) -> List:
        if self.padding_mode == "bottom_right":
            return [{Processings.DetectionBottomRightPadding: {"output_shape": (self.min_height, self.min_width), "pad_value": self.image_pad_value}}]
        elif self.padding_mode == "center":
            return [{Processings.DetectionCenterPadding: {"output_shape": (self.min_height, self.min_width), "pad_value": self.image_pad_value}}]
        else:
            raise RuntimeError(f"KeypointsPadIfNeeded with padding_mode={self.padding_mode} is not implemented.")
