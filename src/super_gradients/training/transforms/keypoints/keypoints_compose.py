from typing import List, Tuple, Optional

import numpy as np

from super_gradients.training.samples import PoseEstimationSample
from .abstract_keypoints_transform import AbstractKeypointTransform


class KeypointsCompose(AbstractKeypointTransform):
    """
    Composes several transforms together
    """

    def __init__(self, transforms: List[AbstractKeypointTransform], load_sample_fn=None):
        """

        :param transforms:         List of keypoint-based transformations
        :param load_sample_fn:     A method to load additional samples if needed (for mixup & mosaic augmentations).
                                   Default value is None, which would raise an error if additional samples are needed.
        """
        for transform in transforms:
            if load_sample_fn is None and transform.additional_samples_count > 0:
                raise RuntimeError(
                    f"Detected transform {transform.__class__.__name__} that require {transform.additional_samples_count} "
                    f"additional samples, but load_sample_fn is None"
                )

        super().__init__()
        self.transforms = transforms
        self.load_sample_fn = load_sample_fn

    def __call__(
        self, image: np.ndarray, mask: np.ndarray, joints: np.ndarray, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply transformation to pose estimation sample passed as a tuple
        This method acts as a wrapper for apply_to_sample method to support old-style API.
        """
        for transform in self.transforms:
            if transform.additional_samples_count > 0:
                raise RuntimeError(f"{transform.__class__.__name__} require additional samples that is not supported in old-style transforms API")

        for t in self.transforms:
            image, mask, joints, areas, bboxes = t(image, mask, joints, areas, bboxes)

        return image, mask, joints, areas, bboxes

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        """
        Applies the series of transformations to the input sample.
        The function may modify the input sample inplace, so input sample should not be used after the call.

        :param sample: Input sample
        :return:       Transformed sample.
        """
        sample = sample.sanitize_sample()
        sample = self._apply_transforms(sample, transforms=self.transforms, load_sample_fn=self.load_sample_fn)
        return sample

    @classmethod
    def _apply_transforms(cls, sample: PoseEstimationSample, transforms: List[AbstractKeypointTransform], load_sample_fn) -> PoseEstimationSample:
        """
        This helper method allows us to query additional samples for mixup & mosaic augmentations
        that would be also passed through augmentation pipeline. Example:

        ```
          transforms:
            - KeypointsBrightnessContrast:
                brightness_range: [ 0.8, 1.2 ]
                contrast_range: [ 0.8, 1.2 ]
                prob: 0.5
            - KeypointsHSV:
                hgain: 20
                sgain: 20
                vgain: 20
                prob: 0.5
            - KeypointsLongestMaxSize:
                max_height: ${dataset_params.image_size}
                max_width: ${dataset_params.image_size}
            - KeypointsMixup:
                prob: ${dataset_params.mixup_prob}
        ```

        In the example above all samples in mixup will be forwarded through KeypointsBrightnessContrast, KeypointsHSV,
        KeypointsLongestMaxSize and only then mixed up.

        :param sample:         Input data sample
        :param transforms:     List of transformations to apply
        :param load_sample_fn: A method to load additional samples if needed
        :return:               A data sample after applying transformations
        """
        applied_transforms_so_far = []
        for t in transforms:
            if not hasattr(t, "additional_samples_count") or t.additional_samples_count == 0:
                sample = t.apply_to_sample(sample)
                applied_transforms_so_far.append(t)
            else:
                additional_samples = [load_sample_fn() for _ in range(t.additional_samples_count)]
                additional_samples = [
                    cls._apply_transforms(
                        sample,
                        applied_transforms_so_far,
                        load_sample_fn=load_sample_fn,
                    )
                    for sample in additional_samples
                ]
                sample.additional_samples = additional_samples
                sample = t.apply_to_sample(sample)

        return sample

    def get_equivalent_preprocessing(self) -> List:
        preprocessing = []
        for t in self.transforms:
            preprocessing += t.get_equivalent_preprocessing()
        return preprocessing

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\t{repr(t)}"
        format_string += "\n)"
        return format_string
