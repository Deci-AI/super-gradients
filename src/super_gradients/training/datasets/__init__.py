from __future__ import absolute_import
import cv2

from super_gradients.training.datasets.data_augmentation import DataAugmentation
from super_gradients.training.datasets.sg_dataset import ListDataset, DirectoryDataSet
from super_gradients.training.datasets.classification_datasets import ImageNetDataset, Cifar10, Cifar100
from super_gradients.training.datasets.detection_datasets import (
    DetectionDataset,
    COCODetectionDataset,
    PascalVOCDetectionDataset,
    YoloDarknetFormatDetectionDataset,
)
from super_gradients.training.datasets.segmentation_datasets.segmentation_dataset import SegmentationDataSet
from super_gradients.training.datasets.segmentation_datasets.pascal_voc_segmentation import (
    PascalVOC2012SegmentationDataSet,
    PascalAUG2012SegmentationDataSet,
    PascalVOCAndAUGUnifiedDataset,
)
from super_gradients.training.datasets.segmentation_datasets.cityscape_segmentation import CityscapesDataset, CityscapesConcatDataset
from super_gradients.training.datasets.segmentation_datasets.coco_segmentation import CoCoSegmentationDataSet
from super_gradients.training.datasets.segmentation_datasets.supervisely_persons_segmentation import SuperviselyPersonsDataset
from super_gradients.training.datasets.pose_estimation_datasets import COCOKeypointsDataset

cv2.setNumThreads(0)


__all__ = [
    "DataAugmentation",
    "ListDataset",
    "DirectoryDataSet",
    "SegmentationDataSet",
    "CityscapesDataset",
    "CityscapesConcatDataset",
    "PascalVOC2012SegmentationDataSet",
    "PascalAUG2012SegmentationDataSet",
    "PascalVOCAndAUGUnifiedDataset",
    "CoCoSegmentationDataSet",
    "DetectionDataset",
    "COCODetectionDataset",
    "YoloDarknetFormatDetectionDataset",
    "PascalVOCDetectionDataset",
    "ImageNetDataset",
    "Cifar10",
    "Cifar100",
    "SuperviselyPersonsDataset",
    "COCOKeypointsDataset",
]
