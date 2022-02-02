from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.datasets.dataset_interfaces import TestDatasetInterface, \
    LibraryDatasetInterface, \
    ClassificationDatasetInterface, Cifar10DatasetInterface, Cifar100DatasetInterface, \
    ImageNetDatasetInterface, TinyImageNetDatasetInterface, \
    CoCoDetectionDatasetInterface, CoCoSegmentationDatasetInterface, CoCo2014DetectionDatasetInterface, \
    PascalAUG2012SegmentationDataSetInterface, PascalVOC2012SegmentationDataSetInterface
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import PascalVOCUnifiedDetectionDataSetInterface, \
    ClassificationTestDatasetInterface, CityscapesDatasetInterface


class DatasetsFactory(BaseFactory):

    def __init__(self):
        type_dict = {
            "classification_test_dataset": ClassificationTestDatasetInterface,
            "library_dataset": LibraryDatasetInterface,
            "classification_dataset": ClassificationDatasetInterface,
            "cifar_10": Cifar10DatasetInterface,
            "cifar_100": Cifar100DatasetInterface,
            "imagenet": ImageNetDatasetInterface,
            "tiny_imagenet": TinyImageNetDatasetInterface,
            "coco2017_detection": CoCoDetectionDatasetInterface,
            "coco2014_detection": CoCo2014DetectionDatasetInterface,
            "coco2017_segmentation": CoCoSegmentationDatasetInterface,
            "pascal_voc_segmentation": PascalVOC2012SegmentationDataSetInterface,
            "pascal_aug_segmentation": PascalAUG2012SegmentationDataSetInterface,
            "pascal_voc_unified_detection": PascalVOCUnifiedDetectionDataSetInterface,
            "cityscapes": CityscapesDatasetInterface,
        }
        super().__init__(type_dict)
