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
    Assemble 4 samples together to make 2x2 grid.
    This transform stacks input samples together to make a square with padding if necessary.
    This transform does not require input samples to have same size.
    If input samples have different sizes (H1,W1), (H2,W2), (H3,W3), (H4,W4), then resulting mosaic will have
    height of max(H1,H2) + max(H3,H4) and width of max(W1+W2, W2+W3), assuming the first sample is located in top left corner,
    second sample is in top right corner, third sample is in bottom left corner and fourth sample is in bottom right corner of mosaic.

    The location of mosaic transform in the transforms list matter.
    It affects what transforms will be applied to all 4 samples.

    In the example below, KeypointsMosaic goes after KeypointsRandomAffineTransform and KeypointsBrightnessContrast.
    This means that all 4 samples will be transformed with KeypointsRandomAffineTransform and KeypointsBrightnessContrast.

    ```yaml
    # This will apply KeypointsRandomAffineTransform and KeypointsBrightnessContrast to four sampls individually
    # and then assemble them together in mosaic
    train_dataset_params:
        transforms:
            - KeypointsRandomAffineTransform:
                min_scale: 0.75
                max_scale: 1.5

            - KeypointsBrightnessContrast:
                brightness_range: [ 0.8, 1.2 ]
                contrast_range: [ 0.8, 1.2 ]
                prob: 0.5

            - KeypointsMosaic:
                prob: 0.5
    ```

    Contrary, if one puts KeypointsMosaic before KeypointsRandomAffineTransform and KeypointsBrightnessContrast,
    then 4 original samples will be assembled in mosaic and then transformed with KeypointsRandomAffineTransform and KeypointsBrightnessContrast:

    ```yaml
    # This will first assemble 4 samples in mosaic and then apply KeypointsRandomAffineTransform and KeypointsBrightnessContrast to the mosaic.
    train_dataset_params:
        transforms:
            - KeypointsRandomAffineTransform:
                min_scale: 0.75
                max_scale: 1.5

            - KeypointsBrightnessContrast:
                brightness_range: [ 0.8, 1.2 ]
                contrast_range: [ 0.8, 1.2 ]
                prob: 0.5

            - KeypointsMosaic:
                prob: 0.5
    ```

    """

    def __init__(self, prob: float, pad_value=(127, 127, 127)):
        """

        :param prob:     Probability to apply the transform.
        :param pad_value Value to pad the image if size of samples does not match.
        """
        super().__init__(additional_samples_count=3)
        self.prob = prob
        self.pad_value = tuple(pad_value)

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        """
        Apply transformation to given estimation sample

        :param sample: A pose estimation sample. The sample must have 3 additional samples in it.
        :return:       A new pose estimation sample that represents the final mosaic.
        """
        if random.random() < self.prob:
            samples = [sample] + sample.additional_samples
            sample = self._apply_mosaic(samples)
        return sample

    def _apply_mosaic(self, samples: List[PoseEstimationSample]) -> PoseEstimationSample:
        """
        Actual method to apply mosaic to the sample.

        :param samples: List of 4 samples to make mosaic from.
        :return:        A new pose estimation sample that represents the final mosaic.
        """
        top_left, top_right, btm_left, btm_right = samples

        mosaic_sample = self._stack_samples_vertically(
            self._stack_samples_horizontally(top_left, top_right, pad_from_top=True), self._stack_samples_horizontally(btm_left, btm_right, pad_from_top=False)
        )

        return mosaic_sample

    def _pad_sample(self, sample: PoseEstimationSample, pad_top: int = 0, pad_left: int = 0, pad_right: int = 0, pad_bottom: int = 0) -> PoseEstimationSample:
        """
        Pad the sample with given padding values.

        :param sample:     Input sample. Sample is modified inplace.
        :param pad_top:    Padding in pixels from top.
        :param pad_left:   Padding in pixels from left.
        :param pad_right:  Padding in pixels from right.
        :param pad_bottom: Padding in pixels from bottom.
        :return:           Modified sample.
        """
        sample.image = cv2.copyMakeBorder(
            sample.image, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right, borderType=cv2.BORDER_CONSTANT, value=self.pad_value
        )
        sample.mask = cv2.copyMakeBorder(sample.mask, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right, borderType=cv2.BORDER_CONSTANT, value=1)

        sample.joints[:, :, 0] += pad_left
        sample.joints[:, :, 1] += pad_top

        sample.bboxes_xywh[:, 0] += pad_left
        sample.bboxes_xywh[:, 1] += pad_top

        return sample

    def _stack_samples_horizontally(self, left: PoseEstimationSample, right: PoseEstimationSample, pad_from_top: bool) -> PoseEstimationSample:
        """
        Stack two samples horizontally.

        :param left:         First sample (Will be located on the left side).
        :param right:        Second sample (Will be location on the right side).
        :param pad_from_top: Controls whether images should be padded from top or from bottom if they have different heights.
        :return:             A stacked sample. If first image has H1,W1 shape and second image has H2,W2 shape,
                             then resulting image will have max(H1,H2), W1+W2 shape.
        """

        max_height = max(left.image.shape[0], right.image.shape[0])
        if pad_from_top:
            left = self._pad_sample(left, pad_top=max_height - left.image.shape[0])
            right = self._pad_sample(right, pad_top=max_height - right.image.shape[0])
        else:
            left = self._pad_sample(left, pad_bottom=max_height - left.image.shape[0])
            right = self._pad_sample(right, pad_bottom=max_height - right.image.shape[0])

        image = np.concatenate([left.image, right.image], axis=1)
        mask = np.concatenate([left.mask, right.mask], axis=1)

        left_sample_width = left.image.shape[1]

        right_bboxes = right.bboxes_xywh
        if right_bboxes is None:
            right_bboxes = np.zeros((0, 4), dtype=np.float32)

        right_joints_offset = np.array([left_sample_width, 0, 0], dtype=right.joints.dtype).reshape((1, 1, 3))
        right_bboxes_offset = np.array([left_sample_width, 0, 0, 0], dtype=right_bboxes.dtype).reshape((1, 4))

        joints = np.concatenate([left.joints, right.joints + right_joints_offset], axis=0)
        bboxes = self._concatenate_arrays(left.bboxes_xywh, right_bboxes + right_bboxes_offset, shape_if_empty=(0, 4))

        is_crowd = np.concatenate([left.is_crowd, right.is_crowd], axis=0)
        areas = self._concatenate_arrays(left.areas, right.areas, shape_if_empty=(0,))
        return PoseEstimationSample(image=image, mask=mask, joints=joints, is_crowd=is_crowd, bboxes_xywh=bboxes, areas=areas, additional_samples=None)

    def _stack_samples_vertically(self, top: PoseEstimationSample, bottom: PoseEstimationSample) -> PoseEstimationSample:
        """
        Stack two samples vertically. If images have different widths, they will be padded to match the width
        of the widest image. In case padding occurs, it will be done from both sides to keep the images centered.

        :param top:    First sample (Will be located on the top).
        :param bottom: Second sample (Will be location on the bottom).
        :return:       A stacked sample. If first image has H1,W1 shape and second image has H2,W2 shape,
                       then resulting image will have H1+H2, max(W1,W2) shape.
        """
        max_width = max(top.image.shape[1], bottom.image.shape[1])

        pad_left = (max_width - top.image.shape[1]) // 2
        pad_right = max_width - top.image.shape[1] - pad_left
        top = self._pad_sample(top, pad_left=pad_left, pad_right=pad_right)

        pad_left = (max_width - bottom.image.shape[1]) // 2
        pad_right = max_width - bottom.image.shape[1] - pad_left
        bottom = self._pad_sample(bottom, pad_left=pad_left, pad_right=pad_right)

        image = np.concatenate([top.image, bottom.image], axis=0)
        mask = np.concatenate([top.mask, bottom.mask], axis=0)

        top_sample_height = top.image.shape[0]

        bottom_bboxes = bottom.bboxes_xywh
        if bottom_bboxes is None:
            bottom_bboxes = np.zeros((0, 4), dtype=np.float32)

        bottom_joints_offset = np.array([0, top_sample_height, 0], dtype=bottom.joints.dtype).reshape((1, 1, 3))
        bottom_bboxes_offset = np.array([0, top_sample_height, 0, 0], dtype=bottom_bboxes.dtype).reshape((1, 4))

        joints = np.concatenate([top.joints, bottom.joints + bottom_joints_offset], axis=0)
        bboxes = self._concatenate_arrays(top.bboxes_xywh, bottom_bboxes + bottom_bboxes_offset, shape_if_empty=(0, 4))

        is_crowd = np.concatenate([top.is_crowd, bottom.is_crowd], axis=0)
        areas = self._concatenate_arrays(top.areas, bottom.areas, shape_if_empty=(0,))
        return PoseEstimationSample(image=image, mask=mask, joints=joints, is_crowd=is_crowd, bboxes_xywh=bboxes, areas=areas, additional_samples=None)

    def _concatenate_arrays(self, arr1: Optional[np.ndarray], arr2: Optional[np.ndarray], shape_if_empty) -> Optional[np.ndarray]:
        """
        Concatenate two arrays. If one of the arrays is None, it will be replaced with array of zeros of given shape.
        This is purely utility function to simplify code of stacking arrays that may be None.
        Arrays must have same number of dims.

        :param arr1:           First array
        :param arr2:           Second array
        :param shape_if_empty: Shape of the array to create if one of the arrays is None.
        :return:               Stacked arrays along first axis. If both arrays are None, then None is returned.
        """
        if arr1 is None and arr2 is None:
            return None
        if arr1 is None:
            arr1 = np.zeros(shape_if_empty, dtype=np.float32)
        if arr2 is None:
            arr2 = np.zeros(shape_if_empty, dtype=np.float32)
        return np.concatenate([arr1, arr2], axis=0)

    def get_equivalent_preprocessing(self):
        raise RuntimeError(f"{self.__class__} does not have equivalent preprocessing because it is non-deterministic.")
