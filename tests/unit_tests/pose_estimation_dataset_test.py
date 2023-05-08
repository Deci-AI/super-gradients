import unittest
import numpy as np
import torch

from super_gradients.training.datasets.pose_estimation_datasets import DEKRTargetsGenerator


class TestPoseEstimationDataset(unittest.TestCase):
    def test_dekr_target_generator(self):
        target_generator = DEKRTargetsGenerator(
            output_stride=4,
            sigma=2,
            center_sigma=4,
            bg_weight=0.1,
            offset_radius=4,
        )

        joints = np.random.randint(0, 255, (4, 17, 3))
        joints[:, :, 2] = 1

        heatmaps, mask, offset_map, offset_weight = target_generator(
            image=torch.zeros((3, 256, 256)),
            joints=joints,
            mask=np.ones((256, 256)),
        )

        self.assertEqual(heatmaps.shape, (18, 64, 64))
        self.assertEqual(mask.shape, (18, 64, 64))
        self.assertEqual(offset_map.shape, (34, 64, 64))
        self.assertEqual(offset_weight.shape, (34, 64, 64))
