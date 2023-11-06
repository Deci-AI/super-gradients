import os
import unittest

import torch
from torch.utils.data import DataLoader

from super_gradients import setup_device
from super_gradients.common import MultiGPUMode
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.cfg_utils import load_dataset_params
from super_gradients.common.object_names import Models
from super_gradients.training import Trainer
from super_gradients.training import models
from super_gradients.training.dataloaders import get_data_loader
from super_gradients.training.datasets import COCOPoseEstimationDataset
from super_gradients.training.metrics import PoseEstimationMetrics

logger = get_logger(__name__)


class YoloNASPoseIntegrationTest(unittest.TestCase):
    """
    YoloNASPoseIntegrationTest - Integration tests for YoloNASPose models
    This test evaluate the performance of trained models on the COCO dataset using
    the modified evaluation protocol that is different from the original COCOEval:

    In COCOEval all images are not resized and pose predictions are evaluated in the resolution of original image
    This is the most accurate and for scientific purposes, this is the way to evaluate the model to report results for publication.

    However this protocol is not suitable for training/validation, because all images has different resolutions and
    it is impossible to collate them to make a batch. So instead we use the following protocol:

    During training/validation, we resize all images to a fixed size (Default is 640x640) using aspect-ratio preserving
    resize of the longest size + padding. Our metric evaluate AP/AR in the resolution of the resized & padded images,
    **not in the resolution of original image**.

    This change has a minor impact on the AP/AR scores while allowing to train/validate the model on a batch of images which
    is much faster than processing images one by one and has no dependency on COCOEval.

    Model            | AP (COCOEval) | AP (Our protocol) |
    -----------------|---------------|-------------------|
    YOLO-NAS POSE N  |         59.68 |             59.68 |
    YOLO-NAS POSE S  |         64.15 |             64.16 |
    YOLO-NAS POSE M  |         67.87 |             67.90 |
    YOLO-NAS POSE L  |         68.24 |             68.28 |

    For evaluation using COCOEval protocol please refer to src/super_gradients/examples/pose_coco_eval/pose_coco_eval.ipynb
    """

    def setUp(self):
        # This is for easy testing on local machine - you can set this environment variable to your own COCO dataset location
        self.data_dir = os.environ.get("SUPER_GRADIENTS_COCO_DATASET_DIR", "/data/coco")
        dataset_params = load_dataset_params("coco_pose_estimation_yolo_nas_dataset_params")
        self.sigmas = dataset_params["oks_sigmas"]
        self.num_joints = dataset_params["num_joints"]

    def _coco2017_val_yolo_nas_pose(self) -> DataLoader:
        loader = get_data_loader(
            config_name="coco_pose_estimation_yolo_nas_dataset_params",
            dataset_cls=COCOPoseEstimationDataset,
            train=False,
            dataset_params=dict(data_dir=self.data_dir),
            dataloader_params=dict(num_workers=0),
        )
        return loader

    def _predict_and_evaluate(self, model, experiment_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        setup_device(device=device, multi_gpu=MultiGPUMode.OFF)
        trainer = Trainer(experiment_name)
        metric = PoseEstimationMetrics(
            post_prediction_callback=model.get_post_prediction_callback(conf=0.01, iou=0.7, post_nms_max_predictions=30),
            num_joints=self.num_joints,
            oks_sigmas=self.sigmas,
        )
        loader = self._coco2017_val_yolo_nas_pose()
        metric_values = trainer.test(model=model, test_loader=loader, test_metrics_list=[metric])
        logger.info(experiment_name, metric_values)
        return metric_values

    def test_yolo_nas_pose_n_coco(self):
        model = models.get(Models.YOLO_NAS_POSE_N, pretrained_weights="coco_pose")
        metric_values = self._predict_and_evaluate(model, "test_yolo_nas_n_coco")
        self.assertAlmostEqual(metric_values["AP"], 0.5968, delta=0.001)

    def test_yolo_nas_s_coco(self):
        model = models.get(Models.YOLO_NAS_POSE_S, num_classes=17, pretrained_weights="coco_pose")
        metric_values = self._predict_and_evaluate(model, "test_yolo_nas_s_coco")
        self.assertAlmostEqual(metric_values["AP"], 0.6416, delta=0.001)

    def test_yolo_nas_m_coco(self):
        model = models.get(Models.YOLO_NAS_POSE_M, pretrained_weights="coco_pose")
        metric_values = self._predict_and_evaluate(model, "test_yolo_nas_m_coco")
        self.assertAlmostEqual(metric_values["AP"], 0.6790, delta=0.001)

    def test_yolo_nas_l_coco(self):
        model = models.get(Models.YOLO_NAS_POSE_L, pretrained_weights="coco_pose")
        metric_values = self._predict_and_evaluate(model, "test_yolo_nas_l_coco")
        self.assertAlmostEqual(metric_values["AP"], 0.6828, delta=0.001)


if __name__ == "__main__":
    unittest.main()
