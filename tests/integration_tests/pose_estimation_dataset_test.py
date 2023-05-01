import os
import unittest

import pkg_resources
import torch
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm

from super_gradients.common.environment.path_utils import normalize_path
from super_gradients.training.dataloaders.dataloaders import _process_dataset_params, get_data_loader
from super_gradients.training.datasets.pose_estimation_datasets import COCOKeypointsDataset
from super_gradients.training import models
from super_gradients.training.metrics import PoseEstimationMetrics
from super_gradients.training.models.pose_estimation_models.dekr_hrnet import DEKRWrapper, DEKRHorisontalFlipWrapper
from super_gradients.training.utils import DEKRPoseEstimationDecodeCallback
from super_gradients.training.utils.pose_estimation import RescoringPoseEstimationDecodeCallback


class PoseEstimationDatasetIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.oks_sigmas = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 1.007, 1.007, 0.087, 0.087, 0.089, 0.089]
        self.flip_indexes_heatmap = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17]
        self.flip_indexes_offset = [
            0,
            2,
            1,
            4,
            3,
            6,
            5,
            8,
            7,
            10,
            9,
            12,
            11,
            14,
            13,
            16,
            15,
        ]

    def test_datasets_instantiation(self):
        GlobalHydra.instance().clear()
        sg_recipes_dir = pkg_resources.resource_filename("super_gradients.recipes", "")
        dataset_config = os.path.join("dataset_params", "coco_pose_estimation_dekr_dataset_params")
        with initialize_config_dir(config_dir=normalize_path(sg_recipes_dir), version_base="1.2"):
            # config is relative to a module
            cfg = compose(config_name=normalize_path(dataset_config))
            train_dataset_params = _process_dataset_params(cfg, dict(), True)
            val_dataset_params = _process_dataset_params(cfg, dict(), True)

            train_dataset = COCOKeypointsDataset(**train_dataset_params)
            assert train_dataset[0] is not None

            val_dataset = COCOKeypointsDataset(**val_dataset_params)
            assert val_dataset[0] is not None

    def test_dataloaders_instantiation(self):
        train_loader = get_data_loader("coco_pose_estimation_dekr_dataset_params", COCOKeypointsDataset, train=True, dataloader_params=dict(num_workers=0))
        val_loader = get_data_loader("coco_pose_estimation_dekr_dataset_params", COCOKeypointsDataset, train=False, dataloader_params=dict(num_workers=0))

        assert next(iter(train_loader)) is not None
        assert next(iter(val_loader)) is not None

    def test_dekr_model(self):
        val_loader = get_data_loader(
            "coco_pose_estimation_dekr_dataset_params",
            COCOKeypointsDataset,
            train=False,
            dataset_params=dict(data_dir="e:/coco2017"),
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
            dataset_params=dict(data_dir="e:/coco2017"),
            dataloader_params=dict(num_workers=0),
        )

        model = models.get("dekr_w32_no_dc", pretrained_weights="coco_pose")
        model = DEKRHorisontalFlipWrapper(model, self.flip_indexes_heatmap, self.flip_indexes_offset, apply_sigmoid=True).cuda().eval()

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
        self.assertAlmostEqual(stats["AP"], 0.6496, delta=0.05)

    def test_dekr_model_with_rescoring(self):
        val_loader = get_data_loader(
            "coco_pose_estimation_dekr_dataset_params",
            COCOKeypointsDataset,
            train=False,
            dataset_params=dict(data_dir="e:/coco2017"),
            dataloader_params=dict(batch_size=1, num_workers=0),
        )

        model = models.get("dekr_w32_no_dc", pretrained_weights="coco_pose")
        model = DEKRHorisontalFlipWrapper(model, self.flip_indexes_heatmap, self.flip_indexes_offset, apply_sigmoid=True).cuda().eval()

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
