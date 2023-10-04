import numpy as np

from typing import List, Tuple, Optional

from super_gradients.common.object_names import Transforms
from super_gradients.common.registry import register_transform
from super_gradients.training.samples import PoseEstimationSample
from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform(Transforms.KeypointsRemoveSmallObjects)
class KeypointsRemoveSmallObjects(AbstractKeypointTransform):
    """
    Remove pose instances from data sample that are too small or have too few visible keypoints.
    """

    def __init__(self, min_visible_keypoints: int = 0, min_instance_area: int = 0, min_bbox_area: int = 0):
        """

        :param min_visible_keypoints: Minimum number of visible keypoints to keep the sample.
                                      Default value is 0 which means that all samples will be kept.
        :param min_instance_area:     Minimum area of instance area to keep the sample
                                      Default value is 0 which means that all samples will be kept.
        :param min_bbox_area:         Minimum area of bounding box to keep the sample
                                      Default value is 0 which means that all samples will be kept.
        """
        super().__init__()
        self.min_visible_keypoints = min_visible_keypoints
        self.min_instance_area = min_instance_area
        self.min_bbox_area = min_bbox_area

    def __call__(
        self, image: np.ndarray, mask: np.ndarray, joints: np.ndarray, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        raise RuntimeError("This transform is not supported for old-style API")

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        """
        Apply transformation to given pose estimation sample.

        :param sample: Input sample to transform.
        :return:       Filtered sample.
        """
        if self.min_visible_keypoints:
            sample = sample.filter_by_visible_joints(self.min_visible_keypoints)
        if self.min_instance_area:
            sample = sample.filter_by_pose_area(self.min_instance_area)
        if self.min_bbox_area:
            sample = sample.filter_by_bbox_area(self.min_bbox_area)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + (
            f"(min_visible_keypoints={self.min_visible_keypoints}, " f"min_instance_area={self.min_instance_area}, " f"min_bbox_area={self.min_bbox_area})"
        )

    def get_equivalent_preprocessing(self) -> List:
        return []
