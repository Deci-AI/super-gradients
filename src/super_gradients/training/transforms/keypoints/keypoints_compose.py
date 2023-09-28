from typing import List, Union

from super_gradients.training.samples import PoseEstimationSample, PoseEstimationSampleFilter
from .abstract_keypoints_transform import AbstractKeypointTransform


class KeypointsCompose:
    def __init__(self, transforms: List[AbstractKeypointTransform], min_bbox_area: Union[int, float], min_visible_joints: int, load_sample_fn):
        """

        :param transforms:
        :param min_bbox_area:
        :param min_visible_joints:
        :param load_sample_fn:
        """
        super().__init__()
        self.transforms = transforms
        self.min_bbox_area = min_bbox_area
        self.load_sample_fn = load_sample_fn
        self.min_visible_joints = min_visible_joints

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        sample = PoseEstimationSampleFilter.sanitize_sample(sample)
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
                sample = t(sample)
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
                sample = t(sample)

            sample = PoseEstimationSampleFilter.filter_by_visible_joints(sample, min_visible_joints)
            sample = PoseEstimationSampleFilter.filter_by_bbox_area(sample, min_bbox_area)

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
