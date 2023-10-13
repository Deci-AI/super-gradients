import unittest

import torch

from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.datasets.pose_estimation_datasets.yolo_nas_pose_collate_fn import (
    flat_collate_tensors_with_batch_index,
    undo_flat_collate_tensors_with_batch_index,
)
from super_gradients.training.losses import YoloNASPoseLoss


class YoloNASPoseTests(unittest.TestCase):
    def test_yolo_nas_pose_forward(self):
        num_joints = 33
        model = models.get(Models.YOLO_NAS_POSE_N, num_classes=num_joints).eval()
        input = torch.randn((1, 3, 640, 640))
        decoded_predictions, _ = model(input)
        pred_bboxes, pred_scores, pred_pose_coords, pred_pose_scores = decoded_predictions

        self.assertEquals(pred_bboxes.shape[2], 4)
        self.assertEquals(pred_scores.shape[2], 1)
        self.assertEquals(pred_pose_coords.shape[2], num_joints)
        self.assertEquals(pred_pose_coords.shape[3], 2)
        self.assertEquals(pred_pose_scores.shape[2], num_joints)

    def test_yolo_nas_pose_loss_function(self):
        model = models.get(Models.YOLO_NAS_POSE_N, num_classes=17)
        input = torch.randn((3, 3, 640, 640))
        outputs = model(input)

        criterion = YoloNASPoseLoss(
            oks_sigmas=[0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089],
        )

        # A single tensor of shape (N, 1 + 4 + Num Joints * 3) (batch_index, x1, y1, x2, y2, [x, y, visibility] * Num Joints)
        # First image has 1 object, second image has 2 objects, third image has no objects

        target_boxes = flat_collate_tensors_with_batch_index(
            [
                torch.tensor([[10, 10, 100, 200]]),
                torch.tensor([[300, 500, 400, 550], [200, 200, 400, 400]]),
                torch.zeros((0, 4)),
            ]
        ).float()

        target_poses = flat_collate_tensors_with_batch_index(
            [
                torch.randn((1, 17, 3)),  # First image has 1 object
                torch.randn((2, 17, 3)),  # Second image has 2 objects
                torch.zeros((0, 17, 3)),  # Third image has no objects
            ]
        ).float()
        target_poses[..., 3] = 2.0  # Mark all joints as visible

        target_crowds = flat_collate_tensors_with_batch_index([torch.zeros((1, 1)), torch.zeros((2, 1)), torch.zeros((0, 1))]).float()

        targets = (target_boxes, target_poses, target_crowds)
        loss = criterion(outputs=outputs, targets=targets)
        loss[0].backward()

    def test_flat_collate_2d(self):
        values = [
            torch.randn([1, 4]),
            torch.randn([2, 4]),
            torch.randn([0, 4]),
            torch.randn([3, 4]),
        ]

        flat_tensor = flat_collate_tensors_with_batch_index(values)
        undo_values = undo_flat_collate_tensors_with_batch_index(flat_tensor, 4)
        assert len(undo_values) == len(values)
        assert (undo_values[0] == values[0]).all()
        assert (undo_values[1] == values[1]).all()
        assert (undo_values[2] == values[2]).all()
        assert (undo_values[3] == values[3]).all()

    def test_flat_collate_3d(self):
        values = [
            torch.randn([1, 17, 3]),
            torch.randn([2, 17, 3]),
            torch.randn([0, 17, 3]),
            torch.randn([3, 17, 3]),
        ]

        flat_tensor = flat_collate_tensors_with_batch_index(values)
        undo_values = undo_flat_collate_tensors_with_batch_index(flat_tensor, 4)
        assert len(undo_values) == len(values)
        assert (undo_values[0] == values[0]).all()
        assert (undo_values[1] == values[1]).all()
        assert (undo_values[2] == values[2]).all()
        assert (undo_values[3] == values[3]).all()

    def test_yolo_nas_pose_replace_classes(self):
        model = models.get(Models.YOLO_NAS_POSE_N, num_classes=17)
        model.replace_head(new_num_classes=20)
        input = torch.randn((1, 3, 640, 640))
        decoded_predictions, _ = model(input)
        pred_bboxes, pred_scores, pred_pose_coords, pred_pose_scores = decoded_predictions

        self.assertEqual(pred_pose_coords.shape[2], 20)
        self.assertEqual(pred_pose_scores.shape[2], 20)


if __name__ == "__main__":
    unittest.main()
