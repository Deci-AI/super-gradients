# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import cv2
import numpy as np


class HeatmapGenerator:
    def __init__(
        self,
        output_res: Tuple[int, int],
        num_joints: int,
        sigma: float,
        center_sigma: float,
        bg_weight: float,
    ):
        """

        :param output_res: (rows, cols)
        :param num_joints:
        """
        self.output_rows, self.output_cols = output_res
        self.num_joints = num_joints
        self.num_joints_with_center = num_joints + 1
        self.sigma = sigma
        self.center_sigma = center_sigma
        self.bg_weight = bg_weight

    def get_heat_val(self, sigma: float, x, y, x0, y0):
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
        return g

    def __call__(self, joints: np.ndarray, mask: np.ndarray):
        """

        :param joints:
        :param mask: [H,W] A mask that indicates which pixels should be included (1) or which one should be excluded (0) from loss computation.
        :return: Tuple of (heatmap, mask, joints, area)
            heatmap - [NumJoints+1, Output Height, Output Width]
            mask -    [NumJoints+1, Output Height, Output Width]
            joints -
            area -
        """
        if joints.shape[1] != self.num_joints_with_center:
            raise RuntimeError(f"The number of joints should be {self.num_joints_with_center}, N keypoints + 1 center joint.")

        heatmaps = np.zeros(
            shape=(self.num_joints_with_center, self.output_rows, self.output_cols),
            dtype=np.float32,
        )

        ignored_hms = 2 * np.ones(
            shape=(self.num_joints_with_center, self.output_rows, self.output_cols),
            dtype=np.float32,
        )  # Start with 2 in all places

        rows, cols = mask.shape  # Rows/Cols corresponds here the original image size

        sx = self.output_cols / cols
        sy = self.output_rows / rows
        joints = joints.copy()
        joints[:, :, 0] *= sx
        joints[:, :, 1] *= sy

        for p in joints:
            for idx, pt in enumerate(p):
                if idx < self.num_joints:  # Last joint index is object center
                    sigma = self.sigma
                else:
                    sigma = self.center_sigma

                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or x >= self.output_cols or y >= self.output_rows:
                        continue

                    ul = int(np.floor(x - 3 * sigma - 1)), int(np.floor(y - 3 * sigma - 1))
                    br = int(np.ceil(x + 3 * sigma + 1)), int(np.ceil(y + 3 * sigma + 1))

                    aa, bb = max(0, ul[1]), min(br[1], self.output_rows)
                    cc, dd = max(0, ul[0]), min(br[0], self.output_cols)

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

        ignored_hms[ignored_hms == 2] = self.bg_weight

        mask = cv2.resize(mask, dsize=(self.output_cols, self.output_rows), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0).astype(np.float32)

        mask = mask * ignored_hms

        return heatmaps, mask, joints


class OffsetGenerator:
    def __init__(self, output_res: Tuple[int, int], num_joints: int, radius: float):
        """

        :param output_res: (rows, cols)
        :param num_joints:
        :param radius:
        """
        self.num_joints_without_center = num_joints
        self.num_joints_with_center = num_joints + 1
        self.output_rows, self.output_cols = output_res
        self.radius = radius

    def __call__(self, joints, area):
        """

        :param joints: [Persons, Joints, 3]
        :param area:
        :return:
        """

        if joints.shape[1] != self.num_joints_with_center:
            raise RuntimeError(f"The number of joints should be {self.num_joints_with_center}, N keypoints + 1 center joint.")

        offset_map = np.zeros(
            (self.num_joints_without_center * 2, self.output_rows, self.output_cols),
            dtype=np.float32,
        )
        weight_map = np.zeros(
            (self.num_joints_without_center * 2, self.output_rows, self.output_cols),
            dtype=np.float32,
        )
        area_map = np.zeros((self.output_rows, self.output_cols), dtype=np.float32)

        for person_id, p in enumerate(joints):
            ct_x = int(p[-1, 0])
            ct_y = int(p[-1, 1])
            ct_v = int(p[-1, 2])
            if ct_v < 1 or ct_x < 0 or ct_y < 0 or ct_x >= self.output_cols or ct_y >= self.output_rows:
                continue

            for idx, pt in enumerate(p[:-1]):
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or x >= self.output_cols or y >= self.output_rows:
                        continue

                    start_x = max(int(ct_x - self.radius), 0)
                    start_y = max(int(ct_y - self.radius), 0)
                    end_x = min(int(ct_x + self.radius), self.output_cols)
                    end_y = min(int(ct_y + self.radius), self.output_rows)

                    for pos_x in range(start_x, end_x):
                        for pos_y in range(start_y, end_y):
                            offset_x = pos_x - x
                            offset_y = pos_y - y
                            if offset_map[idx * 2, pos_y, pos_x] != 0 or offset_map[idx * 2 + 1, pos_y, pos_x] != 0:
                                if area_map[pos_y, pos_x] < area[person_id]:
                                    continue
                            offset_map[idx * 2, pos_y, pos_x] = offset_x
                            offset_map[idx * 2 + 1, pos_y, pos_x] = offset_y
                            weight_map[idx * 2, pos_y, pos_x] = 1.0 / np.sqrt(area[person_id])
                            weight_map[idx * 2 + 1, pos_y, pos_x] = 1.0 / np.sqrt(area[person_id])
                            area_map[pos_y, pos_x] = area[person_id]

        return offset_map, weight_map
