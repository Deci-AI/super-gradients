import os
import unittest
from pathlib import Path

from super_gradients import Trainer
from super_gradients.training import models
from super_gradients.training.datasets import COCODetectionDataset
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.models import YoloPostPredictionCallback
from super_gradients.training.processing import ReverseImageChannels, DetectionLongestMaxSizeRescale, DetectionBottomRightPadding, ImagePermute
from super_gradients.training.utils.detection_utils import DetectionCollateFN, CrowdDetectionCollateFN
from super_gradients.training import dataloaders


class PreprocessingUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mini_coco_data_dir = str(Path(__file__).parent.parent / "data" / "tinycoco")

    def test_getting_preprocessing_params(self):
        expected_image_processor = {
            "ComposeProcessing": {
                "processings": [
                    "ReverseImageChannels",
                    {"DetectionLongestMaxSizeRescale": {"output_shape": (512, 512)}},
                    {"DetectionLongestMaxSizeRescale": {"output_shape": (512, 512)}},
                    {"DetectionBottomRightPadding": {"output_shape": (512, 512), "pad_value": 114}},
                    {"ImagePermute": {"permutation": (2, 0, 1)}},
                ]
            }
        }

        train_dataset_params = {
            "data_dir": self.mini_coco_data_dir,
            "subdir": "images/train2017",
            "json_file": "instances_train2017.json",
            "cache": False,
            "input_dim": [512, 512],
            "transforms": [
                {"DetectionPaddedRescale": {"input_dim": [512, 512]}},
                {"DetectionTargetsFormatTransform": {"input_dim": [512, 512], "output_format": "LABEL_CXCYWH"}},
            ],
        }
        dataset = COCODetectionDataset(**train_dataset_params)
        preprocessing_params = dataset.get_dataset_preprocessing_params()
        self.assertEqual(len(preprocessing_params["class_names"]), 80)
        self.assertEqual(preprocessing_params["image_processor"], expected_image_processor)
        self.assertEqual(preprocessing_params["iou"], 0.65)
        self.assertEqual(preprocessing_params["conf"], 0.5)

    def test_setting_preprocessing_params_from_validation_set(self):
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
        train_loader = dataloaders.get(dataset=trainset, dataloader_params={"collate_fn": DetectionCollateFN()})

        valset = COCODetectionDataset(**val_dataset_params)
        valid_loader = dataloaders.get(dataset=valset, dataloader_params={"collate_fn": CrowdDetectionCollateFN()})

        trainer = Trainer("test_setting_preprocessing_params_from_validation_set")

        detection_train_params_yolox = {
            "max_epochs": 1,
            "lr_mode": "cosine",
            "cosine_final_lr_ratio": 0.05,
            "warmup_bias_lr": 0.0,
            "warmup_momentum": 0.9,
            "initial_lr": 0.02,
            "loss": "yolox_loss",
            "criterion_params": {"strides": [8, 16, 32], "num_classes": 80},  # output strides of all yolo outputs
            "train_metrics_list": [],
            "valid_metrics_list": [DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(), normalize_targets=True, num_cls=5)],
            "metric_to_watch": "mAP@0.50:0.95",
            "greater_metric_to_watch_is_better": True,
            "average_best_models": False,
        }

        model = models.get("yolox_s", num_classes=80)
        trainer.train(model=model, training_params=detection_train_params_yolox, train_loader=train_loader, valid_loader=valid_loader)
        processing_list = model._image_processor.processings
        self.assertTrue(isinstance(processing_list[0], ReverseImageChannels))
        self.assertTrue(isinstance(processing_list[1], DetectionLongestMaxSizeRescale))
        self.assertTrue(isinstance(processing_list[2], DetectionLongestMaxSizeRescale))
        self.assertTrue(isinstance(processing_list[3], DetectionBottomRightPadding))
        self.assertTrue(isinstance(processing_list[4], ImagePermute))
        self.assertTrue(len(processing_list), 5)
        self.assertEqual(model._default_nms_iou, 0.65)
        self.assertEqual(model._default_nms_conf, 0.5)

    def test_setting_preprocessing_params_from_checkpoint(self):
        model = models.get("yolox_s", num_classes=80)
        self.assertTrue(model._image_processor is None)
        self.assertTrue(model._default_nms_iou is None)
        self.assertTrue(model._default_nms_conf is None)
        self.assertTrue(model._class_names is None)

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
        train_loader = dataloaders.get(dataset=trainset, dataloader_params={"collate_fn": DetectionCollateFN()})

        valset = COCODetectionDataset(**val_dataset_params)
        valid_loader = dataloaders.get(dataset=valset, dataloader_params={"collate_fn": CrowdDetectionCollateFN()})

        trainer = Trainer("save_ckpt_for")

        detection_train_params_yolox = {
            "max_epochs": 1,
            "lr_mode": "cosine",
            "cosine_final_lr_ratio": 0.05,
            "warmup_bias_lr": 0.0,
            "warmup_momentum": 0.9,
            "initial_lr": 0.02,
            "loss": "yolox_loss",
            "criterion_params": {"strides": [8, 16, 32], "num_classes": 80},  # output strides of all yolo outputs
            "train_metrics_list": [],
            "valid_metrics_list": [DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(), normalize_targets=True, num_cls=5)],
            "metric_to_watch": "mAP@0.50:0.95",
            "greater_metric_to_watch_is_better": True,
            "average_best_models": False,
        }

        trainer.train(model=model, training_params=detection_train_params_yolox, train_loader=train_loader, valid_loader=valid_loader)

        model = models.get("yolox_s", num_classes=80, checkpoint_path=os.path.join(trainer.checkpoints_dir_path, "ckpt_best.pth"))
        processing_list = model._image_processor.processings
        self.assertTrue(isinstance(processing_list[0], ReverseImageChannels))
        self.assertTrue(isinstance(processing_list[1], DetectionLongestMaxSizeRescale))
        self.assertTrue(isinstance(processing_list[2], DetectionLongestMaxSizeRescale))
        self.assertTrue(isinstance(processing_list[3], DetectionBottomRightPadding))
        self.assertTrue(isinstance(processing_list[4], ImagePermute))
        self.assertTrue(len(processing_list), 5)
        self.assertEqual(model._default_nms_iou, 0.65)
        self.assertEqual(model._default_nms_conf, 0.5)


if __name__ == "__main__":
    unittest.main()
