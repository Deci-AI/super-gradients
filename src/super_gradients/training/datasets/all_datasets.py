from super_gradients.training.datasets.classification_datasets import Cifar10, Cifar100, ImageNetDataset
from super_gradients.training.datasets.detection_datasets import COCODetectionDataset, DetectionDataset, PascalVOCDetectionDataset
from super_gradients.training.datasets.segmentation_datasets import (
    SegmentationDataSet,
    CoCoSegmentationDataSet,
    PascalAUG2012SegmentationDataSet,
    PascalVOC2012SegmentationDataSet,
    CityscapesDataset,
    SuperviselyPersonsDataset,
    PascalVOCAndAUGUnifiedDataset,
)

ALL_DATASETS = {
    "Cifar10": Cifar10,
    "Cifar100": Cifar100,
    "ImageNetDataset": ImageNetDataset,
    "COCODetectionDataset": COCODetectionDataset,
    "DetectionDataset": DetectionDataset,
    "PascalVOCDetectionDataset": PascalVOCDetectionDataset,
    "SegmentationDataSet": SegmentationDataSet,
    "CoCoSegmentationDataSet": CoCoSegmentationDataSet,
    "PascalAUG2012SegmentationDataSet": PascalAUG2012SegmentationDataSet,
    "PascalVOC2012SegmentationDataSet": PascalVOC2012SegmentationDataSet,
    "CityscapesDataset": CityscapesDataset,
    "SuperviselyPersonsDataset": SuperviselyPersonsDataset,
    "PascalVOCAndAUGUnifiedDataset": PascalVOCAndAUGUnifiedDataset,
}
