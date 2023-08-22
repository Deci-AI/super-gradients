import unittest
from pprint import pprint

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam

from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.dataloaders import get_data_loader
from super_gradients.training.datasets import COCOKeypointsDataset
from super_gradients.training.datasets.pose_estimation_datasets.yolo_nas_pose_target_generator import (
    flat_collate_tensors_with_batch_index,
    undo_flat_collate_tensors_with_batch_index,
)
from super_gradients.training.losses import YoloNASPoseLoss
from super_gradients.training.utils.callbacks import ExtremeBatchPoseEstimationVisualizationCallback


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

    def test_dataloader(self):
        loader = get_data_loader(
            config_name="coco_pose_estimation_yolo_nas_dataset_params",
            dataset_cls=COCOKeypointsDataset,
            train=False,
            dataset_params=dict(data_dir="g:/coco2017"),
            dataloader_params=dict(num_workers=0, batch_size=32),
        )
        dataset = loader.dataset
        edge_links = dataset.edge_links
        edge_colors = dataset.edge_colors
        keypoint_colors = dataset.keypoint_colors

        batch = next(iter(loader))
        images, (boxes, joints), extras = batch

        batch_size = len(images)

        images = ExtremeBatchPoseEstimationVisualizationCallback.universal_undo_preprocessing_fn(images)

        target_joints_unpacked = undo_flat_collate_tensors_with_batch_index(joints, batch_size)

        batch_results = ExtremeBatchPoseEstimationVisualizationCallback.visualize_batch(
            images,
            keypoints=target_joints_unpacked,
            scores=None,
            edge_links=edge_links,
            edge_colors=edge_colors,
            keypoint_colors=keypoint_colors,
        )

        for image in batch_results:
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.show()

        pass


if __name__ == "__main__":
    unittest.main()
