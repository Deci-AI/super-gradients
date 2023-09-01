import hashlib
import os
import unittest
from typing import List

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from super_gradients.common import StrictLoad
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.dataloaders import get_data_loader
from super_gradients.training.datasets import COCOKeypointsDataset
from super_gradients.training.losses import YoloNASPoseLoss
from super_gradients.training.metrics.pose_estimation_metrics import PoseEstimationPredictions, PoseEstimationMetrics
from super_gradients.training.models.pose_estimation_models import YoloNASPose
from super_gradients.training.utils.callbacks import ExtremeBatchPoseEstimationVisualizationCallback


class YoloNASPoseTests(unittest.TestCase):
    # def test_forwarad(self):
    #     model = models.get(Models.YOLO_NAS_POSE_S, num_classes=17)
    #     input = torch.randn((1, 3, 640, 640))
    #     decoded_predictions, raw_predictions = model(input)
    #     pred_bboxes, pred_scores, pred_pose_coords, pred_pose_scores = decoded_predictions
    #     cls_score_list, reg_distri_list, pose_regression_list, anchors, anchor_points, num_anchors_list, stride_tensor = raw_predictions
    #     pass
    #
    # def test_loss_function(self):
    #     model = models.get(Models.YOLO_NAS_POSE_S, num_classes=17)
    #     input = torch.randn((3, 3, 640, 640))
    #     decoded_predictions, raw_predictions = model(input)
    #     pred_bboxes, pred_scores, pred_pose_coords, pred_pose_scores = decoded_predictions
    #     cls_score_list, reg_distri_list, pose_regression_list, anchors, anchor_points, num_anchors_list, stride_tensor = raw_predictions
    #
    #     cls_score_list = torch.nn.Parameter(cls_score_list.detach())
    #     reg_distri_list = torch.nn.Parameter(reg_distri_list.detach())
    #     pose_regression_list = torch.nn.Parameter(pose_regression_list.detach())
    #
    #     optimizable_parameters = [cls_score_list, reg_distri_list, pose_regression_list]
    #     optimizer = Adam(optimizable_parameters, lr=0.01)
    #
    #     criterion = YoloNASPoseLoss(
    #         num_classes=17,
    #         oks_sigmas=torch.tensor([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]),
    #     )
    #
    #     # A single tensor of shape (N, 1 + 4 + Num Joints * 3) (batch_index, x1, y1, x2, y2, [x, y, visibility] * Num Joints)
    #     # First image has 1 object, second image has 2 objects, third image has no objects
    #
    #     target_boxes = flat_collate_tensors_with_batch_index(
    #         [
    #             torch.tensor([[10, 10, 100, 200]]),
    #             torch.tensor([[300, 500, 400, 550], [200, 200, 400, 400]]),
    #             torch.zeros((0, 4)),
    #         ]
    #     ).float()
    #
    #     target_poses = flat_collate_tensors_with_batch_index(
    #         [
    #             torch.randn((1, 17, 3)),  # First image has 1 object
    #             torch.randn((2, 17, 3)),  # Second image has 2 objects
    #             torch.zeros((0, 17, 3)),  # Third image has no objects
    #         ]
    #     ).float()
    #     target_poses[..., 3] = 2.0  # Mark all joints as visible
    #
    #     targets = (target_boxes, target_poses)
    #     for _ in range(100):
    #         optimizer.zero_grad()
    #         raw_predictions = (cls_score_list, reg_distri_list, pose_regression_list, anchors, anchor_points, num_anchors_list, stride_tensor)
    #         loss = criterion(outputs=(decoded_predictions, raw_predictions), targets=targets)
    #         loss[0].backward()
    #         pprint(loss)
    #         optimizer.step()
    #
    # def test_flat_collate_2d(self):
    #     values = [
    #         torch.randn([1, 4]),
    #         torch.randn([2, 4]),
    #         torch.randn([0, 4]),
    #         torch.randn([3, 4]),
    #     ]
    #
    #     flat_tensor = flat_collate_tensors_with_batch_index(values)
    #     undo_values = undo_flat_collate_tensors_with_batch_index(flat_tensor, 4)
    #     assert len(undo_values) == len(values)
    #     assert (undo_values[0] == values[0]).all()
    #     assert (undo_values[1] == values[1]).all()
    #     assert (undo_values[2] == values[2]).all()
    #     assert (undo_values[3] == values[3]).all()
    #
    # def test_flat_collate_3d(self):
    #     values = [
    #         torch.randn([1, 17, 3]),
    #         torch.randn([2, 17, 3]),
    #         torch.randn([0, 17, 3]),
    #         torch.randn([3, 17, 3]),
    #     ]
    #
    #     flat_tensor = flat_collate_tensors_with_batch_index(values)
    #     undo_values = undo_flat_collate_tensors_with_batch_index(flat_tensor, 4)
    #     assert len(undo_values) == len(values)
    #     assert (undo_values[0] == values[0]).all()
    #     assert (undo_values[1] == values[1]).all()
    #     assert (undo_values[2] == values[2]).all()
    #     assert (undo_values[3] == values[3]).all()
    #
    # def test_dataloader(self):
    #     loader = get_data_loader(
    #         config_name="coco_pose_estimation_yolo_nas_dataset_params",
    #         dataset_cls=COCOKeypointsDataset,
    #         train=False,
    #         dataset_params=dict(data_dir="g:/coco2017"),
    #         dataloader_params=dict(num_workers=0, batch_size=32),
    #     )
    #     dataset = loader.dataset
    #     edge_links = dataset.edge_links
    #     edge_colors = dataset.edge_colors
    #     keypoint_colors = dataset.keypoint_colors
    #
    #     batch = next(iter(loader))
    #     images, (boxes, joints), extras = batch
    #
    #     batch_size = len(images)
    #
    #     images = ExtremeBatchPoseEstimationVisualizationCallback.universal_undo_preprocessing_fn(images)
    #
    #     target_joints_unpacked = undo_flat_collate_tensors_with_batch_index(joints, batch_size)
    #     target_bboxes_unpacked = undo_flat_collate_tensors_with_batch_index(boxes, batch_size)
    #     batch_results = ExtremeBatchPoseEstimationVisualizationCallback.visualize_batch(
    #         images,
    #         keypoints=target_joints_unpacked,
    #         scores=None,
    #         edge_links=edge_links,
    #         edge_colors=edge_colors,
    #         keypoint_colors=keypoint_colors,
    #         bboxes=target_bboxes_unpacked,
    #     )
    #
    #     for image in batch_results:
    #         plt.figure(figsize=(10, 10))
    #         plt.imshow(image)
    #         plt.show()
    #
    #     pass

    def test_single_batch_overfit(self):
        batch_size = 56
        num_classes = 17

        loader = get_data_loader(
            config_name="coco_pose_estimation_yolo_nas_dataset_params",
            dataset_cls=COCOKeypointsDataset,
            train=False,
            dataset_params=dict(data_dir="g:/coco2017", include_empty_samples=False),
            dataloader_params=dict(num_workers=0, batch_size=batch_size),
        )
        dataset = loader.dataset
        edge_links = dataset.edge_links
        edge_colors = dataset.edge_colors
        keypoint_colors = dataset.keypoint_colors
        oks_sigmas = np.array([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089])
        batch = next(iter(loader))
        images, (boxes, joints), extras = batch

        images_8u = ExtremeBatchPoseEstimationVisualizationCallback.universal_undo_preprocessing_fn(images)

        classification_loss_types = ["focal", "bce"]
        regression_iou_loss_types = ["giou", "ciou"]
        pose_classification_loss_types = ["bce", "focal"]
        use_cocoeval_formula_types = [True, False]
        use_offset_compensation_types = [True, False]

        hyperparameters_grid = []
        for classification_loss_type in classification_loss_types:
            for regression_iou_loss_type in regression_iou_loss_types:
                for pose_classification_loss_type in pose_classification_loss_types:
                    for use_cocoeval_formula in use_cocoeval_formula_types:
                        for use_offset_compensation in use_offset_compensation_types:
                            hyperparameters_grid.append(
                                dict(
                                    learning_rate=0.0001,
                                    classification_loss_type=classification_loss_type,
                                    regression_iou_loss_type=regression_iou_loss_type,
                                    pose_classification_loss_type=pose_classification_loss_type,
                                    classification_loss_weight=1.0,
                                    iou_loss_weight=2.5,
                                    dfl_loss_weight=0.5,
                                    pose_cls_loss_weight=1.0,
                                    pose_reg_loss_weight=1.0,
                                    use_cocoeval_formula=use_cocoeval_formula,
                                    use_offset_compensation=use_offset_compensation,
                                )
                            )

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        for trial in hyperparameters_grid:
            log_dir = "runs/" + hashlib.md5(str(trial).encode("utf-8")).hexdigest()
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)
            torch.cuda.manual_seed_all(0)

            model: YoloNASPose = models.get(
                Models.YOLO_NAS_POSE_S,
                arch_params=dict(heads=dict(YoloNASPoseNDFLHeads=dict(compensate_grid_cell_offset=trial["use_offset_compensation"]))),
                num_classes=num_classes,
                checkpoint_path="https://sghub.deci.ai/models/yolo_nas_s_coco.pth",
                strict_load=StrictLoad.KEY_MATCHING,
            ).cuda()
            optimizer = Adam(model.parameters(), lr=0.0001)
            loss = YoloNASPoseLoss(
                num_classes=num_classes,
                oks_sigmas=oks_sigmas,
                reg_max=16,
                classification_loss_type=trial["classification_loss_type"],
                regression_iou_loss_type=trial["regression_iou_loss_type"],
                classification_loss_weight=trial["classification_loss_weight"],
                iou_loss_weight=trial["iou_loss_weight"],
                dfl_loss_weight=trial["dfl_loss_weight"],
                pose_cls_loss_weight=trial["pose_cls_loss_weight"],
                pose_reg_loss_weight=trial["pose_reg_loss_weight"],
                use_cocoeval_formula=trial["use_cocoeval_formula"],
                pose_classification_loss_type=trial["pose_classification_loss_type"],
            ).cuda()
            inputs = images.cuda()
            targets = boxes.cuda(), joints.cuda()
            scaler = torch.cuda.amp.GradScaler()
            callback = model.get_post_prediction_callback(conf=0.1, iou=0.7, post_nms_max_predictions=30)
            metric = PoseEstimationMetrics(
                oks_sigmas=oks_sigmas,
                post_prediction_callback=callback,
                num_joints=num_classes,
            )

            global_step = 0
            for epoch in range(50):
                for step in range(100):
                    optimizer.zero_grad(set_to_none=True)

                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = model(inputs)
                        loss_for_backward, loss_components = loss(outputs, targets)

                    scaler.scale(loss_for_backward).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    loss_components_dict = {k: v.item() for k, v in zip(loss.component_names, loss_components)}
                    print(loss_components_dict, "epoch", epoch, "step", step)

                    for k, v in loss_components_dict.items():
                        writer.add_scalar(f"loss/{k}", v, global_step)

                    for param_group_index, param_group in enumerate(optimizer.param_groups):
                        lr = param_group["lr"]
                        writer.add_scalar(f"optimizer/pg_{param_group_index}_lr", lr, global_step)

                    global_step += 1

                # End of "epoch"
                metric.reset()
                metric(outputs, targets, **extras)
                metrics = metric.compute()

                # Reduce LR
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.95

                predictions: List[PoseEstimationPredictions] = callback(outputs)
                batch_results = ExtremeBatchPoseEstimationVisualizationCallback.visualize_batch(
                    images_8u,
                    keypoints=[p.poses for p in predictions],
                    bboxes=[p.bboxes for p in predictions],
                    scores=[p.scores for p in predictions],
                    edge_links=edge_links,
                    edge_colors=edge_colors,
                    keypoint_colors=keypoint_colors,
                )

                for image_index, image in enumerate(batch_results):
                    writer.add_image(f"predictions/{image_index}", batch_results[image_index], global_step, dataformats="HWC")

                for metric_name, metric_value in metrics.items():
                    writer.add_scalar("metrics/" + metric_name, metric_value, global_step)

            # End of training - log final scores and hyperparameters
            writer.add_hparams(
                trial,
                metric_dict=metrics,
                hparam_domain_discrete={
                    "classification_loss_type": ["focal", "bce"],
                    "regression_iou_loss_type": ["giou", "ciou"],
                    "pose_classification_loss_type": ["bce", "focal"],
                    "use_cocoeval_formula": [True, False],
                    "use_offset_compensation": [True, False],
                },
            )


if __name__ == "__main__":
    unittest.main()
