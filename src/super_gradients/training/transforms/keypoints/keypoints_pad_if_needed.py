from typing import List, Iterable

import cv2
import numpy as np

from super_gradients.common.object_names import Transforms, Processings
from super_gradients.common.registry.registry import register_transform
from super_gradients.training.samples import PoseEstimationSample
from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform(Transforms.KeypointsPadIfNeeded)
class KeypointsPadIfNeeded(AbstractKeypointTransform):
    """
    Pad image and mask to ensure that resulting image size is not less than `output_size` (rows, cols).
    Image and mask padded from right and bottom, thus joints remains unchanged.
    """

    def __init__(self, min_height: int, min_width: int, image_pad_value: int, mask_pad_value: float, padding_mode: str = "bottom_right"):
        """

        :param output_size: Desired image size (rows, cols)
        :param image_pad_value: Padding value of image
        :param mask_pad_value: Padding value for mask
        """
        if padding_mode not in ("bottom_right", "center"):
            raise ValueError(f"Unknown padding mode: {padding_mode}. Supported modes: 'bottom_right', 'center'")
        super().__init__()
        self.min_height = min_height
        self.min_width = min_width
        self.image_pad_value = image_pad_value
        self.mask_pad_value = mask_pad_value
        self.padding_mode = padding_mode

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        height, width = sample.image.shape[:2]
        original_dtype = sample.mask.dtype

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

        image_pad_value = tuple(self.image_pad_value) if isinstance(self.image_pad_value, Iterable) else tuple([self.image_pad_value] * sample.image.shape[-1])
        sample.image = cv2.copyMakeBorder(
            sample.image, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right, value=image_pad_value, borderType=cv2.BORDER_CONSTANT
        )

        sample.mask = cv2.copyMakeBorder(
            sample.mask.astype(np.uint8),
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            value=self.mask_pad_value,
            borderType=cv2.BORDER_CONSTANT,
        ).astype(original_dtype)

        sample.joints = self.apply_to_keypoints(sample.joints, pad_left, pad_top)
        if sample.bboxes_xywh is not None:
            sample.bboxes_xywh = self.apply_to_bboxes(sample.bboxes_xywh, pad_left, pad_top)

        return sample

    def apply_to_bboxes(self, bboxes: np.ndarray, pad_left, pad_top):
        bboxes = bboxes.copy()
        bboxes[:, 0] += pad_left
        bboxes[:, 1] += pad_top
        return bboxes

    def apply_to_keypoints(self, keypoints: np.ndarray, pad_left, pad_top):
        keypoints = keypoints.copy()
        keypoints[:, :, 0] += pad_left
        keypoints[:, :, 1] += pad_top
        return keypoints

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(min_height={self.min_height}, "
            f"min_width={self.min_width}, "
            f"image_pad_value={self.image_pad_value}, "
            f"mask_pad_value={self.mask_pad_value}, "
            f"padding_mode={self.padding_mode}, "
            f")"
        )

    def get_equivalent_preprocessing(self) -> List:
        if self.padding_mode == "bottom_right":
            return [{Processings.KeypointsBottomRightPadding: {"output_shape": (self.min_height, self.min_width), "pad_value": self.image_pad_value}}]
        else:
            raise RuntimeError(f"KeypointsPadIfNeeded with padding_mode={self.padding_mode} is not implemented.")
