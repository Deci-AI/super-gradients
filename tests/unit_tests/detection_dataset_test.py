import unittest

import hydra
import pkg_resources

from super_gradients.training.datasets import COCODetectionDataset
from super_gradients.training.exceptions.dataset_exceptions import DatasetValidationException, ParameterMismatchException


class DetectionDatasetTest(unittest.TestCase):
    def test_coco_dataset_creation_with_wrong_classes(self):
        with hydra.initialize_config_dir(config_dir=pkg_resources.resource_filename("super_gradients.recipes", "dataset_params/"), version_base="1.2"):
            cfg = hydra.compose(config_name="coco_detection_dataset_params")

            train_dataset_params = {
                "data_dir": cfg.train_dataset_params.data_dir,
                "subdir": cfg.train_dataset_params.subdir,
                "json_file": cfg.train_dataset_params.json_file,
                "cache": False,
                "input_dim": [512, 512],
                "all_classes_list": ["One", "Two", "Three"],
            }
        with self.assertRaises(DatasetValidationException):
            COCODetectionDataset(**train_dataset_params)

    def test_coco_dataset_creation_with_subset_classes(self):
        with hydra.initialize_config_dir(config_dir=pkg_resources.resource_filename("super_gradients.recipes", "dataset_params/"), version_base="1.2"):
            cfg = hydra.compose(config_name="coco_detection_dataset_params")

            train_dataset_params = {
                "data_dir": cfg.train_dataset_params.data_dir,
                "subdir": cfg.train_dataset_params.subdir,
                "json_file": cfg.train_dataset_params.json_file,
                "cache": False,
                "input_dim": [512, 512],
                "all_classes_list": ["car", "person", "bird"],
            }
        with self.assertRaises(ParameterMismatchException):
            COCODetectionDataset(**train_dataset_params)


if __name__ == "__main__":
    unittest.main()
