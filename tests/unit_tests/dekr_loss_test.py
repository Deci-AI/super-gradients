import unittest

import numpy as np
import torch
from super_gradients.training.losses.dekr_loss import DEKRLoss
from super_gradients.training.datasets.pose_estimation_datasets.target_generators import DEKRTargetsGenerator


class DEKRLossTest(unittest.TestCase):
    def test_dekr_loss(self):
        num_joints = 17
        num_persons = 3
        target_generator = DEKRTargetsGenerator(output_stride=4, sigma=2, center_sigma=4, bg_weight=0.1, offset_radius=4)

        joints = np.random.randint(1, 255, (num_persons, num_joints, 3))
        image = torch.randn((3, 256, 256))
        mask = np.ones((256, 256))
        joints[:, :, 2] = 1  # All visible

        targets = target_generator(image, joints, mask)
        gt_heatmaps, gt_mask, gt_offset_map, gt_offset_weight = targets

        self.assertEqual(
            gt_heatmaps.shape, (num_joints + 1, image.shape[1] // target_generator.output_stride, image.shape[2] // target_generator.output_stride)
        )

        random_predictions = torch.randn(
            (1, num_joints + 1, image.shape[1] // target_generator.output_stride, image.shape[2] // target_generator.output_stride)
        ), torch.randn((1, num_joints * 2, image.shape[1] // target_generator.output_stride, image.shape[2] // target_generator.output_stride))

        targets = (
            torch.from_numpy(gt_heatmaps).unsqueeze(0),
            torch.from_numpy(gt_mask).unsqueeze(0),
            torch.from_numpy(gt_offset_map).unsqueeze(0),
            torch.from_numpy(gt_offset_weight).unsqueeze(0),
        )

        loss = DEKRLoss()
        main_loss, loss_components = loss(random_predictions, targets)
        self.assertEqual(len(loss_components), len(loss.component_names))

        perfect_predictions = targets[0], targets[2]
        main_loss, loss_components = loss(perfect_predictions, targets)
        self.assertEqual(main_loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
