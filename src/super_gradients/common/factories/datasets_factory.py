from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.datasets.dataset_interfaces import LibraryDatasetInterface, ClassificationDatasetInterface, Cifar10DatasetInterface,\
    Cifar100DatasetInterface, ImageNetDatasetInterface, TinyImageNetDatasetInterface, CoCoSegmentationDatasetInterface,\
    PascalAUG2012SegmentationDataSetInterface, PascalVOC2012SegmentationDataSetInterface
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import \
    ClassificationTestDatasetInterface, CityscapesDatasetInterface, CoCoDetectionDatasetInterface


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
            "coco2017_segmentation": CoCoSegmentationDatasetInterface,
            "pascal_voc_segmentation": PascalVOC2012SegmentationDataSetInterface,
            "pascal_aug_segmentation": PascalAUG2012SegmentationDataSetInterface,
            "cityscapes": CityscapesDatasetInterface,
        }
        super().__init__(type_dict)
