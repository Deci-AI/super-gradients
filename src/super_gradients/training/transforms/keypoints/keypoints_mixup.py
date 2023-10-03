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
            sample = self.apply_mixup(sample, other)
        return sample

    def apply_mixup(self, sample: PoseEstimationSample, other: PoseEstimationSample) -> PoseEstimationSample:
        sample.image = (sample.image * 0.5 + other.image * 0.5).astype(sample.image)
        sample.mask = np.logical_or(sample.mask, other.mask).astype(sample.mask.dtype)
        sample.joints = np.concatenate([sample.joints, other.joints], axis=0)
        sample.is_crowd = np.concatenate([sample.is_crowd, other.is_crowd], axis=0)

        sample.bboxes = self._concatenate_arrays(sample.bboxes, other.bboxes, (0, 4))
        sample.areas = self._concatenate_arrays(sample.areas, other.areas, (0,))
        sample.additional_samples = None
        return sample

    def _concatenate_arrays(self, arr1: Optional[np.ndarray], arr2: Optional[np.ndarray], shape_if_empty):
        if arr1 is None:
            arr1 = np.zeros(shape_if_empty, dtype=np.float32)
        if arr2 is None:
            arr2 = np.zeros(shape_if_empty, dtype=np.float32)
        return np.concatenate([arr1, arr2], axis=0)

    def get_equivalent_preprocessing(self):
        raise RuntimeError(f"{self.__class__} does not have equivalent preprocessing because it is non-deterministic.")
