import unittest

import super_gradients
from super_gradients.training.datasets import PascalVOCDetectionDataset, COCODetectionDataset
from super_gradients.training.transforms import DetectionMosaic, DetectionPaddedRescale, DetectionTargetsFormatTransform
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat


class SubclassingIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        super_gradients.init_trainer()
        self.batch_size = 64
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

        self.pascal_class_inclusion_lists = [['airplane', 'bicycle'],
                                             ['bird', 'boat', 'bottle', 'bus'],
                                             ['potted plant'],
                                             ['person']]
        self.dataset_parcoco_base_config = dict(data_dir="/data/coco",
                                                subdir="images/val2017",
                                                json_file="instances_val2017.json",
                                                input_dim=(640, 640),
                                                transforms=transforms)

    def test_multiple_pascal_dataset_subclass_before_transforms(self):
        """Run test_pascal_dataset_subclass on multiple inclusion lists"""
        for class_inclusion_list in self.pascal_class_inclusion_lists:
            dataset = PascalVOCDetectionDataset(class_inclusion_list=class_inclusion_list, **self.pascal_base_config)
            dataset.plot(max_samples_per_plot=16, n_plots=1, plot_transformed_data=False)

    def test_multiple_pascal_dataset_subclass_after_transforms(self):
        """Run test_pascal_dataset_subclass on multiple inclusion lists"""
        for class_inclusion_list in self.pascal_class_inclusion_lists:
            dataset = PascalVOCDetectionDataset(class_inclusion_list=class_inclusion_list, **self.pascal_base_config)
            dataset.plot(max_samples_per_plot=16, n_plots=1, plot_transformed_data=True)

    def test_multiple_coco_dataset_subclass_before_transforms(self):
        """Check subclass on multiple inclusions before transform"""
        for class_inclusion_list in self.pascal_class_inclusion_lists:
            dataset = COCODetectionDataset(class_inclusion_list=class_inclusion_list, **self.dataset_parcoco_base_config)
            dataset.plot(max_samples_per_plot=16, n_plots=1, plot_transformed_data=False)

    def test_multiple_coco_dataset_subclass_after_transforms(self):
        """Check subclass on multiple inclusions after transform"""
        for class_inclusion_list in self.pascal_class_inclusion_lists:
            dataset = COCODetectionDataset(class_inclusion_list=class_inclusion_list, **self.dataset_parcoco_base_config)
            dataset.plot(max_samples_per_plot=16, n_plots=1, plot_transformed_data=True)

    def test_non_existing_class(self):
        """Check that EmptyDatasetException is raised when unknown label """
        with self.assertRaises(ValueError):
            PascalVOCDetectionDataset(class_inclusion_list=["new_class"], **self.pascal_base_config)



if __name__ == '__main__':
    unittest.main()
