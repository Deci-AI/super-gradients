import random

import numpy as np
from super_gradients.common.registry import register_transform

from .abstract_obb_transform import AbstractOBBDetectionTransform
from .obb_sample import OBBSample


@register_transform()
class OBBDetectionMixup(AbstractOBBDetectionTransform):
    """
    Apply mixup augmentation and combine two samples into one.
    Images are averaged with equal weights. Targets are concatenated without any changes.
    This transform requires both samples have the same image size. The easiest way to achieve this is to use resize + padding before this transform:

    NOTE: For efficiency, the decision whether to apply the transformation is done (per call) at `get_number_of_additional_samples`

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
        super().__init__()
        self.prob = prob

    def get_number_of_additional_samples(self) -> int:
        do_mixup = random.random() < self.prob
        return int(do_mixup)

    @property
    def may_require_additional_samples(self) -> bool:
        return True

    def apply_to_sample(self, sample: OBBSample) -> OBBSample:
        """
        Apply the transform to a single sample.

        :param sample: An input sample. It should have one additional sample in `additional_samples` field.
        :return:       A new pose estimation sample that represents the mixup sample.
        """
        if sample.additional_samples is not None and len(sample.additional_samples) > 0:
            other = sample.additional_samples[0]
            if sample.image.shape != other.image.shape:
                raise RuntimeError(
                    f"OBBDetectionMixup requires both samples to have the same image shape. "
                    f"Got {sample.image.shape} and {other.image.shape}. "
                    f"Use OBBDetectionLongestMaxSize and OBBDetectionPadIfNeeded to resize and pad images before this transform."
                )
            sample = self._apply_mixup(sample, other)
        return sample

    def _apply_mixup(self, sample: OBBSample, other: OBBSample) -> OBBSample:
        """
        Apply mixup augmentation to a single sample.
        :param sample: First sample.
        :param other:  Second sample.
        :return:       Mixup sample.
        """
        image = (sample.image * 0.5 + other.image * 0.5).astype(sample.image.dtype)
        rboxes_cxcywhr = np.concatenate([sample.rboxes_cxcywhr, other.rboxes_cxcywhr], axis=0)
        labels = np.concatenate([sample.labels, other.labels], axis=0)
        is_crowd = np.concatenate([sample.is_crowd, other.is_crowd], axis=0)

        return OBBSample(image=image, rboxes_cxcywhr=rboxes_cxcywhr, labels=labels, is_crowd=is_crowd, additional_samples=None)

    def get_equivalent_preprocessing(self):
        raise RuntimeError(f"{self.__class__} does not have equivalent preprocessing because it is non-deterministic.")
