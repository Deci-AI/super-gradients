import random
import numpy as np

from typing import Optional

from super_gradients.common.registry import register_transform
from super_gradients.training.samples import PoseEstimationSample

from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform()
class KeypointsMixup(AbstractKeypointTransform):
    """
    Apply mixup augmentation and combine two samples into one.
    Images are averaged with equal weights. Targets are concatenated without any changes.
    This transform requires both samples have the same image size. The easiest way to achieve this is to use resize + padding before this transform:

    ```yaml
    # This will apply KeypointsLongestMaxSize and KeypointsPadIfNeeded to two samples individually
    # and then apply KeypointsMixup to get a single sample.
    train_dataset_params:
        transforms:
            - KeypointsLongestMaxSize:
                max_height: ${dataset_params.image_size}
                max_width: ${dataset_params.image_size}

            - KeypointsPadIfNeeded:
                min_height: ${dataset_params.image_size}
                min_width: ${dataset_params.image_size}
                image_pad_value: [127, 127, 127]
                mask_pad_value: 1
                padding_mode: center

            - KeypointsMixup:
                prob: 0.5
    ```

    :param prob:            Probability to apply the transform.
    """

    def __init__(self, prob: float):
        """

        :param prob:            Probability to apply the transform.
        """
        super().__init__(additional_samples_count=1)
        self.prob = prob

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        """
        Apply the transform to a single sample.

        :param sample: An input sample. It should have one additional sample in `additional_samples` field.
        :return:       A new pose estimation sample that represents the mixup sample.
        """
        if random.random() < self.prob:
            other = sample.additional_samples[0]
            if sample.image.shape != other.image.shape:
                raise RuntimeError(
                    f"KeypointsMixup requires both samples to have the same image shape. "
                    f"Got {sample.image.shape} and {other.image.shape}. "
                    f"Use KeypointsLongestMaxSize and KeypointsPadIfNeeded to resize and pad images before this transform."
                )
            sample = self._apply_mixup(sample, other)
        return sample

    def _apply_mixup(self, sample: PoseEstimationSample, other: PoseEstimationSample) -> PoseEstimationSample:
        """
        Apply mixup augmentation to a single sample.
        :param sample: First sample.
        :param other:  Second sample.
        :return:       Mixup sample.
        """
        image = (sample.image * 0.5 + other.image * 0.5).astype(sample.image.dtype)
        mask = np.logical_or(sample.mask, other.mask).astype(sample.mask.dtype)
        joints = np.concatenate([sample.joints, other.joints], axis=0)
        is_crowd = np.concatenate([sample.is_crowd, other.is_crowd], axis=0)

        bboxes = self._concatenate_arrays(sample.bboxes_xywh, other.bboxes_xywh, (0, 4))
        areas = self._concatenate_arrays(sample.areas, other.areas, (0,))
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
