import unittest

import torch
from tqdm import tqdm

from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import get_data_loader
from super_gradients.training.datasets.pose_estimation_datasets import COCOKeypointsDataset
from super_gradients.training.metrics import PoseEstimationMetrics
from super_gradients.training.models.pose_estimation_models.dekr_hrnet import DEKRWrapper, DEKRHorisontalFlipWrapper
from super_gradients.training.utils import DEKRPoseEstimationDecodeCallback
from super_gradients.training.utils.pose_estimation import RescoringPoseEstimationDecodeCallback


class PoseEstimationModelsIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.oks_sigmas = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 1.007, 1.007, 0.087, 0.087, 0.089, 0.089]
        self.flip_indexes = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    def test_dekr_model(self):
        val_loader = get_data_loader(
            "coco_pose_estimation_dekr_dataset_params",
            COCOKeypointsDataset,
            train=False,
            dataloader_params=dict(num_workers=0),
        )

        model = models.get("dekr_w32_no_dc", pretrained_weights="coco_pose")
        model = DEKRWrapper(model, apply_sigmoid=True).cuda().eval()

        post_prediction_callback = DEKRPoseEstimationDecodeCallback(
            output_stride=4, max_num_people=30, apply_sigmoid=False, keypoint_threshold=0.05, nms_threshold=0.05, nms_num_threshold=8
        )

        post_prediction_callback.apply_sigmoid = False

        metric = PoseEstimationMetrics(
            post_prediction_callback=post_prediction_callback,
            max_objects_per_image=post_prediction_callback.max_num_people,
            num_joints=val_loader.dataset.num_joints,
            oks_sigmas=self.oks_sigmas,
        )

        for inputs, targets, extras in tqdm(val_loader):
            with torch.no_grad(), torch.cuda.amp.autocast(True):
                predictions = model(inputs.cuda(non_blocking=True))
                metric.update(predictions, targets, **extras)

        stats = metric.compute()
        self.assertAlmostEqual(stats["AP"], 0.6308, delta=0.05)

    def test_dekr_model_with_tta(self):
        val_loader = get_data_loader(
            "coco_pose_estimation_dekr_dataset_params",
            COCOKeypointsDataset,
            train=False,
            dataloader_params=dict(num_workers=0),
        )

        model = models.get("dekr_w32_no_dc", pretrained_weights="coco_pose")
        model = DEKRHorisontalFlipWrapper(model, self.flip_indexes, apply_sigmoid=True).cuda().eval()

        post_prediction_callback = DEKRPoseEstimationDecodeCallback(
            output_stride=4, max_num_people=30, apply_sigmoid=False, keypoint_threshold=0.05, nms_threshold=0.05, nms_num_threshold=8
        )

        metric = PoseEstimationMetrics(
            post_prediction_callback=post_prediction_callback,
            max_objects_per_image=post_prediction_callback.max_num_people,
            num_joints=val_loader.dataset.num_joints,
            oks_sigmas=self.oks_sigmas,
        )

        for inputs, targets, extras in tqdm(val_loader):
            with torch.no_grad(), torch.cuda.amp.autocast(True):
                predictions = model(inputs.cuda(non_blocking=True))
                metric.update(predictions, targets, **extras)

        stats = metric.compute()
        self.assertAlmostEqual(stats["AP"], 0.6490, delta=0.05)

    def test_dekr_model_with_rescoring(self):
        val_loader = get_data_loader(
            "coco_pose_estimation_dekr_dataset_params",
            COCOKeypointsDataset,
            train=False,
            dataloader_params=dict(batch_size=1, num_workers=0),
        )

        model = models.get("dekr_w32_no_dc", pretrained_weights="coco_pose")
        model = DEKRHorisontalFlipWrapper(model, self.flip_indexes, apply_sigmoid=True).cuda().eval()

        rescoring = models.get("pose_rescoring_coco", pretrained_weights="coco_pose").cuda().eval()

        post_prediction_callback = DEKRPoseEstimationDecodeCallback(
            output_stride=4, max_num_people=30, apply_sigmoid=False, keypoint_threshold=0.05, nms_threshold=0.05, nms_num_threshold=8
        )

        metric = PoseEstimationMetrics(
            post_prediction_callback=RescoringPoseEstimationDecodeCallback(apply_sigmoid=True),
            max_objects_per_image=post_prediction_callback.max_num_people,
            num_joints=val_loader.dataset.num_joints,
            oks_sigmas=self.oks_sigmas,
        )

        for inputs, targets, extras in tqdm(val_loader):
            with torch.no_grad(), torch.cuda.amp.autocast(True):
                predictions = model(inputs.cuda(non_blocking=True))

                [all_poses], _ = post_prediction_callback(predictions)
                all_poses, new_scores = rescoring(torch.tensor(all_poses).cuda())

                metric.update(preds=(all_poses.unsqueeze(0), new_scores.unsqueeze(0)), target=targets, **extras)

        stats = metric.compute()
        self.assertAlmostEqual(stats["AP"], 0.6734, delta=0.05)


if __name__ == "__main__":
    unittest.main()
