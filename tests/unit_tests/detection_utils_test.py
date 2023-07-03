import os
import unittest
from pathlib import Path

import numpy as np
import torch.cuda

from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, utils as core_utils, models, dataloaders
from super_gradients.training.dataloaders.dataloaders import coco2017_val
from super_gradients.training.datasets import COCODetectionDataset
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.metrics import DetectionMetrics, DetectionMetrics_050
from super_gradients.training.models.detection_models.yolo_base import YoloXPostPredictionCallback
from super_gradients.training.utils.detection_utils import DetectionVisualization, DetectionCollateFN, CrowdDetectionCollateFN
from tests.core_test_utils import is_data_available


class TestDetectionUtils(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.get(Models.YOLOX_N, pretrained_weights="coco").to(self.device)
        self.model.eval()

    @unittest.skipIf(not is_data_available(), "run only when /data is available")
    def test_visualization(self):

        valid_loader = coco2017_val(dataloader_params={"batch_size": 16, "num_workers": 0})
        trainer = Trainer("visualization_test")
        post_prediction_callback = YoloXPostPredictionCallback()

        # Simulate one iteration of validation subset
        batch_i, batch = 0, next(iter(valid_loader))
        imgs, targets = batch[:2]
        imgs = core_utils.tensor_container_to_device(imgs, self.device)
        targets = core_utils.tensor_container_to_device(targets, self.device)
        output = self.model(imgs)
        output = post_prediction_callback(output)
        # Visualize the batch
        DetectionVisualization.visualize_batch(imgs, output, targets, batch_i, COCO_DETECTION_CLASSES_LIST, trainer.checkpoints_dir_path)

        # Assert images ware created and delete them
        img_name = "{}/{}_{}.jpg"
        for i in range(4):
            img_path = img_name.format(trainer.checkpoints_dir_path, batch_i, i)
            self.assertTrue(os.path.exists(img_path))
            os.remove(img_path)

    @unittest.skipIf(not is_data_available(), "run only when /data is available")
    def test_detection_metrics(self):

        valid_loader = coco2017_val(dataloader_params={"batch_size": 16, "num_workers": 0})

        metrics = [
            DetectionMetrics(num_cls=80, post_prediction_callback=YoloXPostPredictionCallback(), normalize_targets=True),
            DetectionMetrics_050(num_cls=80, post_prediction_callback=YoloXPostPredictionCallback(), normalize_targets=True),
            DetectionMetrics(num_cls=80, post_prediction_callback=YoloXPostPredictionCallback(conf=2), normalize_targets=True),
        ]

        ref_values = [
            np.array([0.24701539, 0.40294355, 0.34654024, 0.28485271]),
            np.array([0.34666198, 0.56854934, 0.5079478, 0.40414381]),
            np.array([0.0, 0.0, 0.0, 0.0]),
        ]

        for met, ref_val in zip(metrics, ref_values):
            met.reset()
            for i, (imgs, targets, extras) in enumerate(valid_loader):
                if i > 5:
                    break
                imgs = core_utils.tensor_container_to_device(imgs, self.device)
                targets = core_utils.tensor_container_to_device(targets, self.device)
                output = self.model(imgs)
                met.update(output, targets, device=self.device, inputs=imgs)
            results = met.compute()
            values = np.array([x.item() for x in list(results.values())])
            self.assertTrue(np.allclose(values, ref_val, rtol=1e-3, atol=1e-4))

    @unittest.skipIf(not is_data_available(), "run only when /data is available")
    def test_detection_metrics_with_classwise_ap(self):
        self.mini_coco_data_dir = str(Path(__file__).parent.parent / "data" / "tinycoco")

        model = models.get("yolox_s", pretrained_weights="coco", num_classes=80)

        train_dataset_params = {
            "data_dir": self.mini_coco_data_dir,
            "subdir": "images/train2017",
            "json_file": "instances_train2017.json",
            "cache": False,
            "input_dim": [329, 320],
            "transforms": [
                {"DetectionPaddedRescale": {"input_dim": [512, 512]}},
                {"DetectionTargetsFormatTransform": {"input_dim": [512, 512], "output_format": "LABEL_CXCYWH"}},
            ],
            "with_crowd": False,
        }

        val_dataset_params = {
            "data_dir": self.mini_coco_data_dir,
            "subdir": "images/val2017",
            "json_file": "instances_val2017.json",
            "cache": False,
            "input_dim": [329, 320],
            "transforms": [
                {"DetectionPaddedRescale": {"input_dim": [512, 512]}},
                {"DetectionTargetsFormatTransform": {"input_dim": [512, 512], "output_format": "LABEL_CXCYWH"}},
            ],
        }
        trainset = COCODetectionDataset(**train_dataset_params)
        train_loader = dataloaders.get(dataset=trainset, dataloader_params={"collate_fn": DetectionCollateFN(), "batch_size": 16})

        valset = COCODetectionDataset(**val_dataset_params)
        valid_loader = dataloaders.get(dataset=valset, dataloader_params={"collate_fn": CrowdDetectionCollateFN(), "batch_size": 16})

        trainer = Trainer("test_detection_metrics_with_classwise_ap")

        detection_train_params_yolox = {
            "max_epochs": 5,
            "lr_mode": "cosine",
            "cosine_final_lr_ratio": 0.05,
            "warmup_bias_lr": 0.0,
            "warmup_momentum": 0.9,
            "initial_lr": 0.02,
            "loss": "yolox_loss",
            "mixed_precision": True,
            "criterion_params": {"strides": [8, 16, 32], "num_classes": 80},  # output strides of all yolo outputs
            "train_metrics_list": [],
            "valid_metrics_list": [
                DetectionMetrics(
                    post_prediction_callback=YoloXPostPredictionCallback(),
                    normalize_targets=True,
                    num_cls=80,
                    include_classwise_ap=True,
                    class_names=COCO_DETECTION_CLASSES_LIST,
                    calc_best_score_thresholds=False,
                )
            ],
            "metric_to_watch": "AP@0.50:0.95_car",
            "greater_metric_to_watch_is_better": True,
            "average_best_models": False,
        }

        trainer.train(model=model, training_params=detection_train_params_yolox, train_loader=train_loader, valid_loader=valid_loader)


if __name__ == "__main__":
    unittest.main()
