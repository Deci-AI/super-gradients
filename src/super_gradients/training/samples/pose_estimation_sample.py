import dataclasses
import numpy as np
from typing import Optional, List, Union

__all__ = ["PoseEstimationSample"]

import torch

from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy, xyxy_to_xywh


@dataclasses.dataclass
class PoseEstimationSample:
    """
    :attr image: Input image in [H,W,C] format
    :attr mask: Target mask in [H,W] format
    :attr joints: Target joints in [NumInstances, NumJoints, 3] format. Last dimension contains (x,y,visibility) for each joint.
    :attr areas: (Optional) Numpy array of [N] shape with area of each instance.
    Note this is not a bbox area, but area of the object itself.
    One may use a heuristic `0.53 * box area` as object area approximation if this is not provided.
    :attr bboxes: (Optional) Numpy array of [N,4] shape with bounding box of each instance (XYWH)
    :attr additional_samples: (Optional) List of additional samples for the same image.
    :attr is_crowd: (Optional) Numpy array of [N] shape with is_crowd flag for each instance
    """

    image: np.ndarray
    mask: np.ndarray
    joints: np.ndarray
    areas: Optional[np.ndarray]
    bboxes: Optional[np.ndarray]
    is_crowd: Optional[np.ndarray]
    additional_samples: List["PoseEstimationSample"] = dataclasses.field(default_factory=list)

    @classmethod
    def compute_area_of_joints_bounding_box(self, joints: np.ndarray) -> np.ndarray:
        """
        Compute area of a bounding box for each instance.
        :param joints:  [Num Instances, Num Joints, 3]
        :return: [Num Instances]
        """
        w = np.max(joints[:, :, 0], axis=-1) - np.min(joints[:, :, 0], axis=-1)
        h = np.max(joints[:, :, 1], axis=-1) - np.min(joints[:, :, 1], axis=-1)
        return w * h

    def sanitize_sample(self) -> "PoseEstimationSample":
        """
        Apply sanity checks on the pose sample, which includes:
        - Clamp bbox coordinates to ensure they are within image boundaries
        - Update visibility status of keypoints if they are outside of image boundaries
        - Update area if bbox clipping occurs
        This function does not remove instances, but may make them subject for removal instead.
        :return: self
        """

        if torch.is_tensor(self.image):
            _, image_height, image_width = self.image.shape
        else:
            image_height, image_width, _ = self.image.shape

        # Update joints visibility status
        outside_left = self.joints[:, :, 0] < 0
        outside_top = self.joints[:, :, 1] < 0
        outside_right = self.joints[:, :, 0] >= image_width
        outside_bottom = self.joints[:, :, 1] >= image_height

        outside_image_mask = outside_left | outside_top | outside_right | outside_bottom
        self.joints[outside_image_mask, 2] = 0

        if self.bboxes is not None:
            # Clamp bboxes to image boundaries
            clamped_boxes = xywh_to_xyxy(self.bboxes, image_shape=(image_height, image_width))
            clamped_boxes[..., [0, 2]] = np.clip(clamped_boxes[..., [0, 2]], 0, image_width - 1)
            clamped_boxes[..., [1, 3]] = np.clip(clamped_boxes[..., [1, 3]], 0, image_height - 1)
            clamped_boxes = xyxy_to_xywh(clamped_boxes, image_shape=(image_height, image_width))

            # Recompute sample areas if they are present
            if self.areas is not None:
                area_reduction_factor = clamped_boxes[..., 2:4].prod(axis=-1) / (self.bboxes[..., 2:4].prod(axis=-1) + 1e-6)
                self.areas = self.areas * area_reduction_factor

            self.bboxes = clamped_boxes
        return self

    def filter_by_mask(self, mask: np.ndarray) -> "PoseEstimationSample":
        """
        Remove pose instances with respect to given mask.

        :remark: This is main method to modify instances of the sample.
        If you are implementing a subclass of PoseEstimationSample and adding extra field associated with each pose
        instance (Let's say you add a distance property for each pose from the camera), then you should override
        this method to do filtering on extra attribute as well.

        :param sample: Instance of PoseEstimationSample to modify. Modification done in-place.
        :param mask:   A boolean or integer mask of samples to keep for given sample
        :return:       self
        """
        self.joints = self.joints[mask]
        self.is_crowd = self.is_crowd[mask]
        if self.bboxes is not None:
            self.bboxes = self.bboxes[mask]
        if self.areas is not None:
            self.areas = self.areas[mask]
        return self

    def filter_by_visible_joints(self, min_visible_joints) -> "PoseEstimationSample":
        """
        Remove instances from the sample which has less than N visible joints
        :param min_visible_joints: A minimal number of visible joints a pose has to have in order to be kept.
        :return: self
        """
        visible_joints_mask = self.joints[:, :, 2] > 0
        keep_mask = np.sum(visible_joints_mask, axis=-1) >= min_visible_joints
        return self.filter_by_mask(keep_mask)

    def filter_by_bbox_area(self, min_bbox_area: Union[int, float] = 1) -> "PoseEstimationSample":
        """
        Remove pose instances that has area of the corresponding bounding box less than a certain threshold.

        :param sample:        Instance of PoseEstimationSample to modify. Modification done in-place.
        :param min_bbox_area: Minimal bounding box area of the pose to keep.
        :return:              self
        """
        if self.bboxes is None:
            area = self.compute_area_of_joints_bounding_box(self.joints)
        else:
            area = self.bboxes[..., 2:4].prod(axis=-1)

        keep_mask = area > min_bbox_area
        return self.filter_by_mask(keep_mask)

    def filter_by_pose_area(self, min_instance_area: Union[int, float] = 1) -> "PoseEstimationSample":
        """
        Remove pose instances that has all keypoints marked as invisible or those,
        which area is less than given threshold

        :param sample:            Instance of PoseEstimationSample to modify. Modification done in-place.
        :param min_instance_area: Minimal area of the pose to keep.
        :return:                  self
        """

        visible_joints_mask = self.joints[:, :, 2] > 0
        keep_mask = np.sum(visible_joints_mask, axis=-1) > 0

        # Filter instances with too small area
        if min_instance_area > 0:
            if self.areas is None:
                areas = self.compute_area_of_joints_bounding_box(self.joints)
            else:
                areas = self.areas

            keep_area_mask = areas > min_instance_area
            keep_mask &= keep_area_mask

        return self.filter_by_mask(keep_mask)
