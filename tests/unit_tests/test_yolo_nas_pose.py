import unittest
from pprint import pprint
from typing import List

import torch
from torch import Tensor
from torch.optim import Adam

from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.losses import YoloNASPoseLoss


class YoloNASPoseTests(unittest.TestCase):
    def test_forwarad(self):
        model = models.get(Models.YOLO_NAS_POSE_S, num_classes=17)
        input = torch.randn((1, 3, 640, 640))
        decoded_predictions, raw_predictions = model(input)
        pred_bboxes, pred_scores, pred_pose_coords, pred_pose_scores = decoded_predictions
        cls_score_list, reg_distri_list, pose_regression_list, anchors, anchor_points, num_anchors_list, stride_tensor = raw_predictions
        pass

    def test_loss_function(self):
        model = models.get(Models.YOLO_NAS_POSE_S, num_classes=17)
        input = torch.randn((3, 3, 640, 640))
        decoded_predictions, raw_predictions = model(input)
        pred_bboxes, pred_scores, pred_pose_coords, pred_pose_scores = decoded_predictions
        cls_score_list, reg_distri_list, pose_regression_list, anchors, anchor_points, num_anchors_list, stride_tensor = raw_predictions

        cls_score_list = torch.nn.Parameter(cls_score_list.detach())
        reg_distri_list = torch.nn.Parameter(reg_distri_list.detach())
        pose_regression_list = torch.nn.Parameter(pose_regression_list.detach())

        optimizable_parameters = [cls_score_list, reg_distri_list, pose_regression_list]
        optimizer = Adam(optimizable_parameters, lr=0.01)

        criterion = YoloNASPoseLoss(
            num_classes=17,
            oks_sigmas=torch.tensor([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 1.007, 1.007, 0.087, 0.087, 0.089, 0.089]),
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

        mask = None

        targets = (target_boxes, target_poses, mask)
        for _ in range(100):
            optimizer.zero_grad()
            raw_predictions = (cls_score_list, reg_distri_list, pose_regression_list, anchors, anchor_points, num_anchors_list, stride_tensor)
            loss = criterion(outputs=(decoded_predictions, raw_predictions), targets=targets)
            loss[0].backward()
            pprint(loss)
            optimizer.step()


def flat_collate_tensors_with_batch_index(labels_batch: List[Tensor]) -> Tensor:
    """
    Stack a batch id column to targets and concatenate
    :param labels_batch: a list of targets per image (each of arbitrary length: [N1, ..., C], [N2, ..., C], [N3, ..., C],...)
    :return: A single tensor of shape [N1+N2+N3+..., ..., C+1], where N is the total number of targets in a batch
             and the 1st column is batch item index
    """
    labels_batch_indexed = []
    for i, labels in enumerate(labels_batch):
        batch_column = labels.new_ones(labels.shape[:-1] + (1,)) * i
        labels = torch.cat((batch_column, labels), dim=-1)
        labels_batch_indexed.append(labels)
    return torch.cat(labels_batch_indexed, 0)


if __name__ == "__main__":
    unittest.main()
