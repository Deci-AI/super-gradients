import unittest

from typing import List
import super_gradients
import torch
from super_gradients.training.datasets import PascalVOCDetectionDataSetV2, COCODetectionDatasetV2
from super_gradients.training.utils.detection_utils import DetectionCollateFN
import os

from super_gradients.training.transforms import DetectionPaddedRescale, DetectionTargetsFormatTransform
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat


class SubclassingIntegrationTest(unittest.TestCase):

    def setUp(self) -> None:
        super_gradients.init_trainer()
        self.BATCH_SIZE = 64
        self.PASCAL_CLASS_INCLUSION_LISTS = [['aeroplane', 'bicycle'],
                                             ['bird', 'boat', 'bottle', 'bus'],
                                             ['pottedplant']]

    def test_multiple_pascal_dataset_subclass(self):
        for class_inclusion_list in self.PASCAL_CLASS_INCLUSION_LISTS:
            self.test_pascal_dataset_subclass(class_inclusion_list)

    def test_pascal_dataset_subclass(self, class_inclusion_list: List[str]):
        """Plot a single image with single bbox of an object from the sub class list, when in mosaic mode.
        :param class_inclusion_list:  List of sub class names (from coco classes).
        """

        transforms = [DetectionPaddedRescale(input_dim=(640, 640), max_targets=120),
                      DetectionTargetsFormatTransform(output_format=DetectionTargetsFormat.XYXY_LABEL)]

        PascalVOCDetectionDataSetV2(
            data_dir='/home/louis.dupont/data/pascal_unified_coco_format/',
            images_sub_directory='images/train2012/',
            input_dim=(640, 640),
            transforms=transforms,
            class_inclusion_list=class_inclusion_list
        ).plot(max_samples_per_plot=16, n_plots=1, plot_transformed_data=False)


if __name__ == '__main__':
    unittest.main()
