import tempfile
import os
import unittest
from typing import Dict, Union, Any

import numpy as np
import pkg_resources
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from pydantic.main import deepcopy

import super_gradients
from super_gradients.training.dataloaders.dataloaders import _process_dataset_params
from super_gradients.training.datasets import PascalVOCDetectionDataset, COCODetectionDataset
from super_gradients.training.transforms import DetectionMosaic, DetectionPaddedRescale, DetectionTargetsFormatTransform
from super_gradients.training.datasets.data_formats.default_formats import XYXY_LABEL
from super_gradients.training.exceptions.dataset_exceptions import EmptyDatasetException
from super_gradients.common.environment.path_utils import normalize_path


class COCODetectionDataset6Channels(COCODetectionDataset):
    def get_sample(self, index: int) -> Dict[str, Union[np.ndarray, Any]]:
        img = self.get_resized_image(index)
        img = np.concatenate((img, img), 2)
        annotation = deepcopy(self.annotations[index])
        return {"image": img, **annotation}


class DatasetIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        super_gradients.init_trainer()
        self.batch_size = 64
        self.max_samples_per_plot = 16
        self.n_plot = 1
        transforms = [
            DetectionMosaic(input_dim=(640, 640), prob=0.8),
            DetectionPaddedRescale(input_dim=(640, 640)),
            DetectionTargetsFormatTransform(input_dim=(640, 640), output_format=XYXY_LABEL),
        ]

        self.test_dir = tempfile.TemporaryDirectory().name
        PascalVOCDetectionDataset.download(self.test_dir)
        self.pascal_class_inclusion_lists = [["aeroplane", "bicycle"], ["bird", "boat", "bottle", "bus"], ["pottedplant"], ["person"]]
        self.pascal_base_config = dict(data_dir=self.test_dir, images_sub_directory="images/train2012/", input_dim=(640, 640), transforms=transforms)

        self.coco_class_inclusion_lists = [["airplane", "bicycle"], ["bird", "boat", "bottle", "bus"], ["potted plant"], ["person"]]
        self.dataset_coco_base_config = dict(
            data_dir="/data/coco",
            subdir="images/val2017",
            json_file="instances_val2017.json",
            input_dim=(640, 640),
            transforms=transforms,
        )

    def test_multiple_pascal_dataset_subclass_before_transforms(self):
        """Run test_pascal_dataset_subclass on multiple inclusion lists"""
        for class_inclusion_list in self.pascal_class_inclusion_lists:
            dataset = PascalVOCDetectionDataset(
                class_inclusion_list=class_inclusion_list, max_num_samples=self.max_samples_per_plot * self.n_plot, **self.pascal_base_config
            )
            dataset.plot(max_samples_per_plot=self.max_samples_per_plot, n_plots=self.n_plot, plot_transformed_data=False)

    def test_multiple_pascal_dataset_subclass_after_transforms(self):
        """Run test_pascal_dataset_subclass on multiple inclusion lists"""
        for class_inclusion_list in self.pascal_class_inclusion_lists:
            dataset = PascalVOCDetectionDataset(
                class_inclusion_list=class_inclusion_list, max_num_samples=self.max_samples_per_plot * self.n_plot, **self.pascal_base_config
            )
            dataset.plot(max_samples_per_plot=self.max_samples_per_plot, n_plots=self.n_plot, plot_transformed_data=True)

    def test_multiple_coco_dataset_subclass_before_transforms(self):
        """Check subclass on multiple inclusions before transform"""
        for class_inclusion_list in self.coco_class_inclusion_lists:
            dataset = COCODetectionDataset(
                class_inclusion_list=class_inclusion_list, max_num_samples=self.max_samples_per_plot * self.n_plot, **self.dataset_coco_base_config
            )
            dataset.plot(max_samples_per_plot=self.max_samples_per_plot, n_plots=self.n_plot, plot_transformed_data=False)

    def test_multiple_coco_dataset_subclass_after_transforms(self):
        """Check subclass on multiple inclusions after transform"""
        for class_inclusion_list in self.coco_class_inclusion_lists:
            dataset = COCODetectionDataset(
                class_inclusion_list=class_inclusion_list, max_num_samples=self.max_samples_per_plot * self.n_plot, **self.dataset_coco_base_config
            )
            dataset.plot(max_samples_per_plot=self.max_samples_per_plot, n_plots=self.n_plot, plot_transformed_data=True)

    def test_subclass_non_existing_class(self):
        """Check that EmptyDatasetException is raised when unknown label."""
        with self.assertRaises(ValueError):
            PascalVOCDetectionDataset(class_inclusion_list=["new_class"], **self.pascal_base_config)

    def test_sub_sampling_dataset(self):
        """Check that sub sampling works."""

        full_dataset = PascalVOCDetectionDataset(**self.pascal_base_config)

        with self.assertRaises(EmptyDatasetException):
            PascalVOCDetectionDataset(max_num_samples=0, **self.pascal_base_config)

        for max_num_samples in [1, 10, 1000, 1_000_000]:
            sampled_dataset = PascalVOCDetectionDataset(max_num_samples=max_num_samples, **self.pascal_base_config)
            self.assertEqual(len(sampled_dataset), min(max_num_samples, len(full_dataset)))

    def test_detection_dataset_transforms_with_unique_channel_count(self):
        GlobalHydra.instance().clear()
        sg_recipes_dir = pkg_resources.resource_filename("super_gradients.recipes", "")
        dataset_config = os.path.join("dataset_params", "coco_detection_dataset_params")
        with initialize_config_dir(config_dir=normalize_path(sg_recipes_dir), version_base="1.2"):
            # config is relative to a module
            cfg = compose(config_name=normalize_path(dataset_config))
            dataset_params = _process_dataset_params(cfg, dict(), True)

        coco_base_recipe_transforms = dataset_params["transforms"]
        dataset_config = deepcopy(self.dataset_coco_base_config)
        dataset_config["transforms"] = coco_base_recipe_transforms
        dataset = COCODetectionDataset6Channels(**dataset_config)
        self.assertEqual(dataset.__getitem__(0)[0].shape[0], 6)


if __name__ == "__main__":
    unittest.main()
