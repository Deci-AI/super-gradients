import random
import cv2
import numpy as np

from typing import Optional, List

from super_gradients.common.registry import register_transform
from super_gradients.training.samples import PoseEstimationSample
from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform()
class KeypointsMosaic(AbstractKeypointTransform):
    """
    Mix two samples together.

    :attr prob:            Probability to apply the transform.
    :attr hgain:           Hue gain.
    :attr sgain:           Saturation gain.
    :attr vgain:           Value gain.
    """

    def __init__(self, prob: float, pad_value=(127, 127, 127)):
        """

        :param prob:            Probability to apply the transform.
        :param hgain:           Hue gain.
        :param sgain:           Saturation gain.
        :param vgain:           Value gain.
        """
        super().__init__(additional_samples_count=3)
        self.prob = prob
        self.pad_value = tuple(pad_value)

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        if random.random() < self.prob:
            sample = self.apply_mosaic(sample, sample.additional_samples)
        return sample

    def apply_mosaic(self, sample: PoseEstimationSample, other: List[PoseEstimationSample]) -> PoseEstimationSample:
        top_left = sample
        top_right = other[0]
        btm_left = other[1]
        btm_right = other[2]

        mosaic_sample = self.stack_samples_vertically(
            self.stack_samples_horisontally(top_left, top_right, pad_from_top=True), self.stack_samples_horisontally(btm_left, btm_right, pad_from_top=False)
        )

        return mosaic_sample

    def pad_sample(self, sample: PoseEstimationSample, pad_top=0, pad_left=0, pad_right=0, pad_bottom=0):
        sample.image = cv2.copyMakeBorder(
            sample.image, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right, borderType=cv2.BORDER_CONSTANT, value=self.pad_value
        )
        sample.mask = cv2.copyMakeBorder(sample.mask, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right, borderType=cv2.BORDER_CONSTANT, value=1)

        sample.joints[:, :, 0] += pad_left
        sample.joints[:, :, 1] += pad_top

        sample.bboxes[:, 0] += pad_left
        sample.bboxes[:, 1] += pad_top

        return sample

    def stack_samples_horisontally(self, left, right, pad_from_top):
        max_height = max(left.image.shape[0], right.image.shape[0])
        if pad_from_top:
            left = self.pad_sample(left, pad_top=max_height - left.image.shape[0])
            right = self.pad_sample(right, pad_top=max_height - right.image.shape[0])
        else:
            left = self.pad_sample(left, pad_bottom=max_height - left.image.shape[0])
            right = self.pad_sample(right, pad_bottom=max_height - right.image.shape[0])

        image = np.concatenate([left.image, right.image], axis=1)
        mask = np.concatenate([left.mask, right.mask], axis=1)

        left_sample_width = left.image.shape[1]

        right_bboxes = right.bboxes
        if right_bboxes is None:
            right_bboxes = np.zeros((0, 4), dtype=np.float32)

        right_joints_offset = np.array([left_sample_width, 0, 0], dtype=right.joints.dtype).reshape((1, 1, 3))
        right_bboxes_offset = np.array([left_sample_width, 0, 0, 0], dtype=right_bboxes.dtype).reshape((1, 4))

        joints = np.concatenate([left.joints, right.joints + right_joints_offset], axis=0)
        bboxes = self._concatenate_arrays(left.bboxes, right_bboxes + right_bboxes_offset, shape_if_empty=(0, 4))

        is_crowd = np.concatenate([left.is_crowd, right.is_crowd], axis=0)
        areas = self._concatenate_arrays(left.areas, right.areas, shape_if_empty=(0,))
        return PoseEstimationSample(image=image, mask=mask, joints=joints, is_crowd=is_crowd, bboxes=bboxes, areas=areas)

    def stack_samples_vertically(self, top, bottom):
        max_width = max(top.image.shape[1], bottom.image.shape[1])

        pad_left = (max_width - top.image.shape[1]) // 2
        pad_right = max_width - top.image.shape[1] - pad_left
        top = self.pad_sample(top, pad_left=pad_left, pad_right=pad_right)

        pad_left = (max_width - bottom.image.shape[1]) // 2
        pad_right = max_width - bottom.image.shape[1] - pad_left
        bottom = self.pad_sample(bottom, pad_left=pad_left, pad_right=pad_right)

        image = np.concatenate([top.image, bottom.image], axis=0)
        mask = np.concatenate([top.mask, bottom.mask], axis=0)

        top_sample_height = top.image.shape[0]

        bottom_bboxes = bottom.bboxes
        if bottom_bboxes is None:
            bottom_bboxes = np.zeros((0, 4), dtype=np.float32)

        bottom_joints_offset = np.array([0, top_sample_height, 0], dtype=bottom.joints.dtype).reshape((1, 1, 3))
        bottom_bboxes_offset = np.array([0, top_sample_height, 0, 0], dtype=bottom_bboxes.dtype).reshape((1, 4))

        joints = np.concatenate([top.joints, bottom.joints + bottom_joints_offset], axis=0)
        bboxes = self._concatenate_arrays(top.bboxes, bottom_bboxes + bottom_bboxes_offset, shape_if_empty=(0, 4))

        is_crowd = np.concatenate([top.is_crowd, bottom.is_crowd], axis=0)
        areas = self._concatenate_arrays(top.areas, bottom.areas, shape_if_empty=(0,))
        return PoseEstimationSample(image=image, mask=mask, joints=joints, is_crowd=is_crowd, bboxes=bboxes, areas=areas)

    def _concatenate_arrays(self, arr1: Optional[np.ndarray], arr2: Optional[np.ndarray], shape_if_empty):
        if arr1 is None:
            arr1 = np.zeros(shape_if_empty, dtype=np.float32)
        if arr2 is None:
            arr2 = np.zeros(shape_if_empty, dtype=np.float32)
        return np.concatenate([arr1, arr2], axis=0)

    def get_equivalent_preprocessing(self):
        raise RuntimeError(f"{self.__class__} does not have equivalent preprocessing because it is non-deterministic.")
