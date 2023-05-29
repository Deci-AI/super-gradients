import unittest
from pathlib import Path

from super_gradients.training.dataloaders import coco2017_train_yolo_nas
from super_gradients.training.datasets import COCODetectionDataset
from super_gradients.training.datasets.data_formats.default_formats import LABEL_CXCYWH
from super_gradients.training.exceptions.dataset_exceptions import DatasetValidationException, ParameterMismatchException
from super_gradients.training.transforms import DetectionMosaic, DetectionTargetsFormatTransform, DetectionPaddedRescale


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


if __name__ == "__main__":
    unittest.main()
