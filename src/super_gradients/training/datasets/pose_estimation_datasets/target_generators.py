import abc
from typing import Tuple, Dict, Union

import cv2
import numpy as np
from torch import Tensor

from super_gradients.common.registry.registry import register_target_generator

__all__ = ["KeypointsTargetsGenerator", "DEKRTargetsGenerator"]


class KeypointsTargetsGenerator:
    @abc.abstractmethod
    def __call__(self, image: Tensor, joints: np.ndarray, mask: np.ndarray) -> Union[Tensor, Tuple[Tensor, ...], Dict[str, Tensor]]:
        """
        Encode input joints into target tensors

        :param image: [C,H,W] Input image tensor
        :param joints: [Num Instances, Num Joints, 3] Last channel represents (x, y, visibility)
        :param mask: [H,W] Mask representing valid image areas. For instance, in COCO dataset crowd targets
                           are not used during training and corresponding instances will be zero-masked.
                           Your implementation may use this mask when generating targets.
        :return: Encoded targets
        """
        raise NotImplementedError()


@register_target_generator()
class DEKRTargetsGenerator(KeypointsTargetsGenerator):
    """
    Target generator for pose estimation task tailored for the DEKR paper (https://arxiv.org/abs/2104.02300)
    """

    def __init__(self, output_stride: int, sigma: float, center_sigma: float, bg_weight: float, offset_radius: float):
        """

        :param output_stride: Downsampling factor for target maps (w.r.t to input image resolution)
        :param sigma: Sigma of the gaussian kernel used to generate the heatmap (Effective radius of the heatmap would be 3*sigma)
        :param center_sigma: Sigma of the gaussian kernel used to generate the instance "center" heatmap (Effective radius of the heatmap would be 3*sigma)
        :param bg_weight: Weight assigned to all background pixels (used to re-weight the heatmap loss)
        :param offset_radius: Radius for the offset encoding (in pixels)
        """
        self.output_stride = output_stride
        self.sigma = sigma
        self.center_sigma = center_sigma
        self.bg_weight = bg_weight
        self.offset_radius = offset_radius

    def get_heat_val(self, sigma: float, x, y, x0, y0) -> float:
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
        return g

    def compute_area(self, joints: np.ndarray) -> np.ndarray:
        """
        Compute area of a bounding box for each instance
        :param joints:  [Num Instances, Num Joints, 3]
        :return: [Num Instances]
        """
        w = np.max(joints[:, :, 0], axis=-1) - np.min(joints[:, :, 0], axis=-1)
        h = np.max(joints[:, :, 1], axis=-1) - np.min(joints[:, :, 1], axis=-1)
        return w * h

    def sort_joints_by_area(self, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rearrange joints in descending order of area of bounding box around them
        """
        area = self.compute_area(joints)
        order = np.argsort(-area)
        joints = joints[order]
        area = area[order]
        return joints, area

    def augment_with_center_joint(self, joints: np.ndarray) -> np.ndarray:
        """
        Augment set of joints with additional center joint.
        Returns a new array with shape [Instances, Joints+1, 3] where the last joint is the center joint.
        Only instances with at least one visible joint are returned.

        :param joints: [Num Instances, Num Joints, 3] Last channel represents (x, y, visibility)
        :return: [Num Instances, Num Joints + 1, 3]
        """
        augmented_joints = []
        num_joints = joints.shape[1]
        num_joints_with_center = num_joints + 1

        for keypoints in joints:
            # Computing a center point for each person
            visible_keypoints = keypoints[:, 2] > 0
            joints_sum = np.sum(keypoints[:, :2] * np.expand_dims(visible_keypoints, -1), axis=0)
            num_vis_joints = np.count_nonzero(visible_keypoints)
            if num_vis_joints == 0:
                raise ValueError("No visible joints found in instance. ")

            keypoints_with_center = np.zeros((num_joints_with_center, 3))
            keypoints_with_center[0:num_joints] = keypoints
            keypoints_with_center[-1, :2] = joints_sum / num_vis_joints
            keypoints_with_center[-1, 2] = 1

            augmented_joints.append(keypoints_with_center)

        joints = np.array(augmented_joints, dtype=np.float32).reshape((-1, num_joints_with_center, 3))
        return joints

    def __call__(self, image: Tensor, joints: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode the keypoints into dense targets that participate in loss computation.
        :param image: Image tensor [3, H, W]
        :param joints: [Instances, NumJoints, 3]
        :param mask: [H,W] A mask that indicates which pixels should be included (1) or which one should be excluded (0) from loss computation.
        :return: Tuple of (heatmap, mask, offset, offset_weight)
            heatmap    - [NumJoints+1, H // Output Stride, W // Output Stride]
            mask       - [NumJoints+1, H // Output Stride, H // Output Stride]
            offset     - [NumJoints*2, H // Output Stride, W // Output Stride]
            offset_weight - [NumJoints*2, H // Output Stride, W // Output Stride]
        """
        if image.shape[1:3] != mask.shape[:2]:
            raise ValueError(f"Image and mask should have the same shape {image.shape[1:3]} != {mask.shape[:2]}")

        if image.shape[1] % self.output_stride != 0 or image.shape[2] % self.output_stride != 0:
            raise ValueError("Image shape should be divisible by output stride")

        num_instances, num_joints, _ = joints.shape
        num_joints_with_center = num_joints + 1

        joints, area = self.sort_joints_by_area(joints)
        joints = self.augment_with_center_joint(joints)

        # Compute the size of the target maps
        rows, cols = mask.shape
        output_rows, output_cols = rows // self.output_stride, cols // self.output_stride

        heatmaps = np.zeros(
            shape=(num_joints_with_center, output_rows, output_cols),
            dtype=np.float32,
        )

        ignored_hms = 2 * np.ones(
            shape=(num_joints_with_center, output_rows, output_cols),
            dtype=np.float32,
        )  # Start with 2 in all places

        offset_map = np.zeros(
            (num_joints * 2, output_rows, output_cols),
            dtype=np.float32,
        )
        offset_weight = np.zeros(
            (num_joints * 2, output_rows, output_cols),
            dtype=np.float32,
        )

        sx = output_cols / cols
        sy = output_rows / rows
        joints = joints.copy()
        joints[:, :, 0] *= sx
        joints[:, :, 1] *= sy

        for person_id, p in enumerate(joints):
            for idx, pt in enumerate(p):
                if idx < num_joints:  # Last joint index is object center
                    sigma = self.sigma
                else:
                    sigma = self.center_sigma

                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or x >= output_cols or y >= output_rows:
                        continue

                    ul = int(np.floor(x - 3 * sigma - 1)), int(np.floor(y - 3 * sigma - 1))
                    br = int(np.ceil(x + 3 * sigma + 1)), int(np.ceil(y + 3 * sigma + 1))

                    aa, bb = max(0, ul[1]), min(br[1], output_rows)
                    cc, dd = max(0, ul[0]), min(br[0], output_cols)

                    joint_rg = np.zeros((bb - aa, dd - cc), dtype=np.float32)
                    for sy in range(aa, bb):
                        for sx in range(cc, dd):
                            # EK: Note we round x/y values here to obtain clear peak in the center of odd-sized heatmap
                            # joint_rg[sy - aa, sx - cc] = self.get_heat_val(sigma, sx, sy, x, y)
                            joint_rg[sy - aa, sx - cc] = self.get_heat_val(sigma, sx, sy, int(x), int(y))

                    # It is important for RFL loss to have 1.0 in heatmap. since 0.9999 would be interpreted as negative pixel
                    joint_rg[joint_rg.shape[0] // 2, joint_rg.shape[1] // 2] = 1

                    heatmaps[idx, aa:bb, cc:dd] = np.maximum(heatmaps[idx, aa:bb, cc:dd], joint_rg)
                    # print(heatmaps[-1, 0, 0])
                    ignored_hms[idx, aa:bb, cc:dd] = 1.0

        for person_id, p in enumerate(joints):
            person_area = area[person_id]
            offset_weight_factor = 1.0 / np.clip(np.sqrt(person_area), a_min=1, a_max=None)
            ct_x = int(p[-1, 0])
            ct_y = int(p[-1, 1])
            ct_v = int(p[-1, 2])
            if ct_v < 1 or ct_x < 0 or ct_y < 0 or ct_x >= output_cols or ct_y >= output_rows:
                continue

            for idx, pt in enumerate(p[:-1]):
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or x >= output_cols or y >= output_rows:
                        continue

                    start_x = max(int(ct_x - self.offset_radius), 0)
                    start_y = max(int(ct_y - self.offset_radius), 0)
                    end_x = min(int(ct_x + self.offset_radius), output_cols)
                    end_y = min(int(ct_y + self.offset_radius), output_rows)

                    for pos_x in range(start_x, end_x):
                        for pos_y in range(start_y, end_y):
                            offset_x = pos_x - x
                            offset_y = pos_y - y

                            offset_map[idx * 2, pos_y, pos_x] = offset_x
                            offset_map[idx * 2 + 1, pos_y, pos_x] = offset_y
                            offset_weight[idx * 2, pos_y, pos_x] = offset_weight_factor
                            offset_weight[idx * 2 + 1, pos_y, pos_x] = offset_weight_factor

        ignored_hms[ignored_hms == 2] = self.bg_weight

        mask = cv2.resize(mask, dsize=(output_cols, output_rows), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0).astype(np.float32)
        mask = mask * ignored_hms

        return heatmaps, mask, offset_map, offset_weight
