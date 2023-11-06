import os
import unittest
from pathlib import Path

import numpy as np
import torch
import torchvision as tv

from super_gradients import Trainer
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.factories.processing_factory import ProcessingFactory
from super_gradients.module_interfaces import HasPreprocessingParams
from super_gradients.training import dataloaders
from super_gradients.training import models
from super_gradients.training.datasets import COCODetectionDataset
from super_gradients.training.datasets.classification_datasets.torchvision_utils import get_torchvision_transforms_equivalent_processing
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.models import YoloXPostPredictionCallback
from super_gradients.training.processing import (
    ReverseImageChannels,
    DetectionLongestMaxSizeRescale,
    DetectionBottomRightPadding,
    ImagePermute,
    ComposeProcessing,
)
from super_gradients.training.transforms import DetectionPaddedRescale, DetectionRGB2BGR
from super_gradients.training.utils.collate_fn import DetectionCollateFN, CrowdDetectionCollateFN


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
        self.assertIsInstance(trainset, HasPreprocessingParams)
        train_loader = dataloaders.get(dataset=trainset, dataloader_params={"collate_fn": DetectionCollateFN(), "num_workers": 0})

        valset = COCODetectionDataset(**val_dataset_params)
        self.assertIsInstance(valset, HasPreprocessingParams)
        valid_loader = dataloaders.get(dataset=valset, dataloader_params={"collate_fn": CrowdDetectionCollateFN(), "num_workers": 0})

        trainer = Trainer("test_setting_preprocessing_params_from_validation_set")

        detection_train_params_yolox = {
            "max_epochs": 1,
            "lr_mode": "CosineLRScheduler",
            "cosine_final_lr_ratio": 0.05,
            "warmup_bias_lr": 0.0,
            "warmup_momentum": 0.9,
            "initial_lr": 0.02,
            "loss": "YoloXDetectionLoss",
            "criterion_params": {"strides": [8, 16, 32], "num_classes": 80},  # output strides of all yolo outputs
            "train_metrics_list": [],
            "valid_metrics_list": [DetectionMetrics(post_prediction_callback=YoloXPostPredictionCallback(), normalize_targets=True, num_cls=80)],
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

        checkpoint_path = os.path.join(trainer.checkpoints_dir_path, "ckpt_best.pth")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.assertTrue("processing_params" in checkpoint)

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
            "lr_mode": "CosineLRScheduler",
            "cosine_final_lr_ratio": 0.05,
            "warmup_bias_lr": 0.0,
            "warmup_momentum": 0.9,
            "initial_lr": 0.02,
            "loss": "YoloXDetectionLoss",
            "criterion_params": {"strides": [8, 16, 32], "num_classes": 80},  # output strides of all yolo outputs
            "train_metrics_list": [],
            "valid_metrics_list": [DetectionMetrics(post_prediction_callback=YoloXPostPredictionCallback(), normalize_targets=True, num_cls=80)],
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

        checkpoint_path = os.path.join(trainer.checkpoints_dir_path, "ckpt_best.pth")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.assertTrue("processing_params" in checkpoint)

    def test_processings_from_dataset_params(self):
        transforms = [DetectionRGB2BGR(prob=1), DetectionPaddedRescale(input_dim=(512, 512))]

        processings = []
        for t in transforms:
            processings += t.get_equivalent_preprocessing()

        instantiated_processing = ListFactory(ProcessingFactory()).get(processings)
        processing_pipeline = ComposeProcessing(instantiated_processing)
        result = processing_pipeline.preprocess_image(np.zeros((480, 640, 3)))
        print(result)

    def test_get_torchvision_transforms_equivalent_processing(self):
        from PIL import Image

        tv_transforms = tv.transforms.Compose(
            [
                tv.transforms.Resize(512),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        input = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        expected_output = tv_transforms(Image.fromarray(input)).numpy()

        processing = get_torchvision_transforms_equivalent_processing(tv_transforms)

        instantiated_processing = ListFactory(ProcessingFactory()).get(processing)
        processing_pipeline = ComposeProcessing(instantiated_processing)
        actual_output = processing_pipeline.preprocess_image(input)[0]

        self.assertEqual(actual_output.shape, expected_output.shape)
        np.testing.assert_allclose(actual_output, expected_output, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
