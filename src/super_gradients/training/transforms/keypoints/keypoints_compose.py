from typing import List, Union, Tuple, Optional

import numpy as np

from super_gradients.training.samples import PoseEstimationSample
from .abstract_keypoints_transform import AbstractKeypointTransform
from .keypoints_mixup import KeypointsMixup
from .keypoints_mosaic import KeypointsMosaic


class KeypointsCompose(AbstractKeypointTransform):
    """
    Composes several transforms together
    """

    def __init__(self, transforms: List[AbstractKeypointTransform], min_bbox_area: Union[int, float] = 0, min_visible_joints: int = 0, load_sample_fn=None):
        """

        :param transforms:         List of keypoint-based transformations
        :param min_bbox_area:      Minimal bbox area of the pose instances to keep them.
                                   Default value is 0, which means that all pose instances will be kept.
        :param min_visible_joints: Minimal number of visible joints to keep the pose instance.
                                   Default value is 0, which means that all pose instances will be kept.
        :param load_sample_fn:     A method to load additional samples if needed (for mixup & mosaic augmentations).
                                   Default value is None, which would raise an error if additional samples are needed.
        """
        if load_sample_fn is None and (KeypointsMixup in transforms or KeypointsMosaic in transforms):
            raise RuntimeError("KeyointsMixup & KeypointsMosaic augmentations require load_sample_fn to be passed")

        super().__init__()
        self.transforms = transforms
        self.min_bbox_area = min_bbox_area
        self.load_sample_fn = load_sample_fn
        self.min_visible_joints = min_visible_joints

    def __call__(
        self, image: np.ndarray, mask: np.ndarray, joints: np.ndarray, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply transformation to pose estimation sample passed as a tuple
        This method acts as a wrapper for apply_to_sample method to support old-style API.
        """
        if KeypointsMixup in self.transforms or KeypointsMosaic in self.transforms:
            raise RuntimeError("KeypointsMixup & KeypointsMosaic augmentations are not supported in old-style transforms API")

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
        sample = self._apply_transforms(
            sample, transforms=self.transforms, load_sample_fn=self.load_sample_fn, min_bbox_area=self.min_bbox_area, min_visible_joints=self.min_visible_joints
        )
        return sample

    @classmethod
    def _apply_transforms(
        cls, sample: PoseEstimationSample, transforms: List[AbstractKeypointTransform], load_sample_fn, min_bbox_area, min_visible_joints
    ) -> PoseEstimationSample:
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
        :param min_bbox_area:  Min bbox area of the pose instances to keep them
        :return:               Transformed sample
        """
        applied_transforms_so_far = []
        for t in transforms:
            if t.additional_samples_count == 0:
                sample = t.apply_to_sample(sample)
                applied_transforms_so_far.append(t)
            else:
                additional_samples = [load_sample_fn() for _ in range(t.additional_samples_count)]
                additional_samples = [
                    cls._apply_transforms(
                        sample, applied_transforms_so_far, load_sample_fn=load_sample_fn, min_bbox_area=min_bbox_area, min_visible_joints=min_visible_joints
                    )
                    for sample in additional_samples
                ]
                sample.additional_samples = additional_samples
                sample = t.apply_to_sample(sample)

            sample = sample.filter_by_visible_joints(min_visible_joints)
            sample = sample.filter_by_bbox_area(min_bbox_area)

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
