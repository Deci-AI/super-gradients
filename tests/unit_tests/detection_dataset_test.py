import unittest
from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader

from super_gradients import Trainer
from super_gradients.training import models, dataloaders
from super_gradients.training.dataloaders import coco2017_train_yolo_nas, get_data_loader
from super_gradients.training.datasets import COCODetectionDataset
from super_gradients.training.datasets.data_formats.default_formats import LABEL_CXCYWH
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.common.exceptions.dataset_exceptions import DatasetValidationException, ParameterMismatchException
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.models import YoloXPostPredictionCallback
from super_gradients.training.transforms import DetectionMosaic, DetectionTargetsFormatTransform, DetectionPaddedRescale
from super_gradients.training.utils.detection_utils import DetectionCollateFN, CrowdDetectionCollateFN


class DummyCOCODetectionDatasetInheritor(COCODetectionDataset):
    def __init__(self, json_file: str, subdir: str, dummy_field: int, *args, **kwargs):
        super(DummyCOCODetectionDatasetInheritor, self).__init__(json_file=json_file, subdir=subdir, *args, **kwargs)
        self.dummy_field = dummy_field


def dummy_coco2017_inheritor_train_yolo_nas(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_detection_yolo_nas_dataset_params",
        dataset_cls=DummyCOCODetectionDatasetInheritor,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


class DetectionDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mini_coco_data_dir = str(Path(__file__).parent.parent / "data" / "tinycoco")

    def test_normal_coco_dataset_creation(self):
        train_dataset_params = {
            "data_dir": self.mini_coco_data_dir,
            "subdir": "images/train2017",
            "json_file": "instances_train2017.json",
            "cache": False,
            "input_dim": [512, 512],
        }
        COCODetectionDataset(**train_dataset_params)

    def test_coco_dataset_creation_with_wrong_classes(self):
        train_dataset_params = {
            "data_dir": self.mini_coco_data_dir,
            "subdir": "images/train2017",
            "json_file": "instances_train2017.json",
            "cache": False,
            "input_dim": [512, 512],
            "all_classes_list": ["One", "Two", "Three"],
        }
        with self.assertRaises(DatasetValidationException):
            COCODetectionDataset(**train_dataset_params)

    def test_coco_dataset_creation_with_subset_classes(self):
        train_dataset_params = {
            "data_dir": self.mini_coco_data_dir,
            "subdir": "images/train2017",
            "json_file": "instances_train2017.json",
            "cache": False,
            "input_dim": [512, 512],
            "all_classes_list": ["car", "person", "bird"],
        }
        with self.assertRaises(ParameterMismatchException):
            COCODetectionDataset(**train_dataset_params)

    def test_coco_detection_dataset_override_image_size(self):
        train_dataset_params = {
            "data_dir": self.mini_coco_data_dir,
            "input_dim": [512, 512],
        }
        train_dataloader_params = {"num_workers": 0}
        dataloader = coco2017_train_yolo_nas(dataset_params=train_dataset_params, dataloader_params=train_dataloader_params)
        batch = next(iter(dataloader))
        print(batch[0].shape)
        self.assertEqual(batch[0].shape[2], 512)
        self.assertEqual(batch[0].shape[3], 512)

    def test_coco_detection_dataset_override_image_size_single_scalar(self):
        train_dataset_params = {
            "data_dir": self.mini_coco_data_dir,
            "input_dim": 384,
        }
        train_dataloader_params = {"num_workers": 0}
        dataloader = coco2017_train_yolo_nas(dataset_params=train_dataset_params, dataloader_params=train_dataloader_params)
        batch = next(iter(dataloader))
        print(batch[0].shape)
        self.assertEqual(batch[0].shape[2], 384)
        self.assertEqual(batch[0].shape[3], 384)

    def test_coco_detection_dataset_override_with_objects(self):
        train_dataset_params = {
            "data_dir": self.mini_coco_data_dir,
            "input_dim": 384,
            "transforms": [
                DetectionMosaic(input_dim=384),
                DetectionPaddedRescale(input_dim=384, max_targets=10),
                DetectionTargetsFormatTransform(max_targets=10, output_format=LABEL_CXCYWH),
            ],
        }
        train_dataloader_params = {"num_workers": 0}
        dataloader = coco2017_train_yolo_nas(dataset_params=train_dataset_params, dataloader_params=train_dataloader_params)
        batch = next(iter(dataloader))
        print(batch[0].shape)
        self.assertEqual(batch[0].shape[2], 384)
        self.assertEqual(batch[0].shape[3], 384)

    def test_coco_detection_dataset_override_with_new_entries(self):
        train_dataset_params = {
            "data_dir": self.mini_coco_data_dir,
            "input_dim": 384,
            "transforms": [
                DetectionMosaic(input_dim=384),
                DetectionPaddedRescale(input_dim=384, max_targets=10),
                DetectionTargetsFormatTransform(max_targets=10, output_format=LABEL_CXCYWH),
            ],
            "dummy_field": 10,
        }
        train_dataloader_params = {"num_workers": 0}
        dataloader = dummy_coco2017_inheritor_train_yolo_nas(dataset_params=train_dataset_params, dataloader_params=train_dataloader_params)
        batch = next(iter(dataloader))
        print(batch[0].shape)
        self.assertEqual(batch[0].shape[2], 384)
        self.assertEqual(batch[0].shape[3], 384)
        self.assertEqual(dataloader.dataset.dummy_field, 10)

    def test_coco_detection_metrics_with_classwise_ap(self):
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
            "mixed_precision": False,
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
