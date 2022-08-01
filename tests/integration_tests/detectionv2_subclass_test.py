import unittest

import super_gradients
from super_gradients.training.datasets import PascalVOCDetectionDataSetV2
from super_gradients.training.transforms import DetectionPaddedRescale, DetectionTargetsFormatTransform
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat
from super_gradients.training.exceptions.dataset_exceptions import EmptyDatasetException


class SubclassingIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        super_gradients.init_trainer()
        self.BATCH_SIZE = 64
        self.PASCAL_CLASS_INCLUSION_LISTS = [['aeroplane', 'bicycle'],
                                             ['bird', 'boat', 'bottle', 'bus'],
                                             ['pottedplant'],
                                             ['person']]
        transforms = [DetectionPaddedRescale(input_dim=(640, 640), max_targets=120),
                      DetectionTargetsFormatTransform(output_format=DetectionTargetsFormat.XYXY_LABEL)]
        self.PASCAL_BASE_CONFIG = dict(data_dir='/home/louis.dupont/data/pascal_unified_coco_format/',
                                       images_sub_directory='images/train2012/',
                                       input_dim=(640, 640),
                                       transforms=transforms)


    def test_multiple_pascal_dataset_subclass(self):
        """Run test_pascal_dataset_subclass on multiple inclusion lists"""
        for class_inclusion_list in self.PASCAL_CLASS_INCLUSION_LISTS:
            dataset = PascalVOCDetectionDataSetV2(class_inclusion_list=class_inclusion_list, **self.PASCAL_BASE_CONFIG)
            dataset.plot(max_samples_per_plot=16, n_plots=1, plot_transformed_data=False)

    def test_non_existing_class(self):
        """Run test_pascal_dataset_subclass on multiple inclusion lists"""
        with self.assertRaises(EmptyDatasetException):
            PascalVOCDetectionDataSetV2(class_inclusion_list=["new_class"], **self.PASCAL_BASE_CONFIG)


if __name__ == '__main__':
    unittest.main()
