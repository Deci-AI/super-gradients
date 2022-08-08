import unittest

import super_gradients
from super_gradients.training.datasets import PascalVOCDetectionDataset, COCODetectionDataset
from super_gradients.training.transforms import DetectionMosaic, DetectionPaddedRescale, DetectionTargetsFormatTransform
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat
from super_gradients.training.exceptions.dataset_exceptions import EmptyDatasetException


class DatasetIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        super_gradients.init_trainer()
        self.batch_size = 64
        self.max_samples_per_plot = 16
        self.n_plot = 1
        transforms = [DetectionMosaic(input_dim=(640, 640), prob=0.8),
                      DetectionPaddedRescale(input_dim=(640, 640), max_targets=120),
                      DetectionTargetsFormatTransform(output_format=DetectionTargetsFormat.XYXY_LABEL)]

        self.pascal_class_inclusion_lists = [['aeroplane', 'bicycle'],
                                             ['bird', 'boat', 'bottle', 'bus'],
                                             ['pottedplant'],
                                             ['person']]
        self.pascal_base_config = dict(data_dir='/home/louis.dupont/data/pascal_unified_coco_format/',
                                       images_sub_directory='images/train2012/',
                                       input_dim=(640, 640),
                                       transforms=transforms)

        self.coco_class_inclusion_lists = [['airplane', 'bicycle'],
                                           ['bird', 'boat', 'bottle', 'bus'],
                                           ['potted plant'],
                                           ['person']]
        self.dataset_coco_base_config = dict(data_dir="/data/coco",
                                             subdir="images/val2017",
                                             json_file="instances_val2017.json",
                                             input_dim=(640, 640),
                                             transforms=transforms,)

    def test_multiple_pascal_dataset_subclass_before_transforms(self):
        """Run test_pascal_dataset_subclass on multiple inclusion lists"""
        for class_inclusion_list in self.pascal_class_inclusion_lists:
            dataset = PascalVOCDetectionDataset(class_inclusion_list=class_inclusion_list,
                                                max_num_samples=self.max_samples_per_plot * self.n_plot,
                                                **self.pascal_base_config)
            dataset.plot(max_samples_per_plot=self.max_samples_per_plot, n_plots=self.n_plot, plot_transformed_data=False)

    def test_multiple_pascal_dataset_subclass_after_transforms(self):
        """Run test_pascal_dataset_subclass on multiple inclusion lists"""
        for class_inclusion_list in self.pascal_class_inclusion_lists:
            dataset = PascalVOCDetectionDataset(class_inclusion_list=class_inclusion_list,
                                                max_num_samples=self.max_samples_per_plot * self.n_plot,
                                                **self.pascal_base_config)
            dataset.plot(max_samples_per_plot=self.max_samples_per_plot, n_plots=self.n_plot, plot_transformed_data=True)

    def test_multiple_coco_dataset_subclass_before_transforms(self):
        """Check subclass on multiple inclusions before transform"""
        for class_inclusion_list in self.coco_class_inclusion_lists:
            dataset = COCODetectionDataset(class_inclusion_list=class_inclusion_list,
                                           max_num_samples=self.max_samples_per_plot * self.n_plot,
                                           **self.dataset_coco_base_config)
            dataset.plot(max_samples_per_plot=self.max_samples_per_plot, n_plots=self.n_plot, plot_transformed_data=False)

    def test_multiple_coco_dataset_subclass_after_transforms(self):
        """Check subclass on multiple inclusions after transform"""
        for class_inclusion_list in self.coco_class_inclusion_lists:
            dataset = COCODetectionDataset(class_inclusion_list=class_inclusion_list,
                                           max_num_samples=self.max_samples_per_plot * self.n_plot,
                                           **self.dataset_coco_base_config)
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


if __name__ == '__main__':
    unittest.main()
