from collections import defaultdict
from typing import Dict, List, Type

from super_gradients.training.datasets.dataset_interfaces import DatasetInterface, TestDatasetInterface, \
    LibraryDatasetInterface, \
    ClassificationDatasetInterface, Cifar10DatasetInterface, Cifar100DatasetInterface, \
    ImageNetDatasetInterface, TinyImageNetDatasetInterface, CoCoSegmentationDatasetInterface,\
    PascalAUG2012SegmentationDataSetInterface, PascalVOC2012SegmentationDataSetInterface
from super_gradients.common.data_types.enum.deep_learning_task import DeepLearningTask
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import CoCoDetectionDatasetInterface


# Fixme: THis should be the same as the factory, which is not the case!
class DatasetNames:
    TEST_DATASET = "test_dataset"
    LIBRARY_DATASET = "library_dataset"
    CLASSIFICATION_DATASET = "classification_dataset"
    CIFAR_10 = "cifar_10"
    CIFAR_100 = "cifar_100"
    IMAGENET = "imagenet"
    TINY_IMAGENET = "tiny_imagenet"
    COCO = "coco"
    PASCAL_VOC = "pascal_voc"
    PASCAL_AUG = "pascal_aug"


CLASSIFICATION_DATASETS = {
    DatasetNames.TEST_DATASET: TestDatasetInterface,
    DatasetNames.LIBRARY_DATASET: LibraryDatasetInterface,
    DatasetNames.CLASSIFICATION_DATASET: ClassificationDatasetInterface,
    DatasetNames.CIFAR_10: Cifar10DatasetInterface,
    DatasetNames.CIFAR_100: Cifar100DatasetInterface,
    DatasetNames.IMAGENET: ImageNetDatasetInterface,
    DatasetNames.TINY_IMAGENET: TinyImageNetDatasetInterface
}

OBJECT_DETECTION_DATASETS = {
    DatasetNames.COCO: CoCoDetectionDatasetInterface,
}

SEMANTIC_SEGMENTATION_DATASETS = {
    DatasetNames.COCO: CoCoSegmentationDatasetInterface,
    DatasetNames.PASCAL_VOC: PascalVOC2012SegmentationDataSetInterface,
    DatasetNames.PASCAL_AUG: PascalAUG2012SegmentationDataSetInterface
}


class DataSetDoesNotExistException(Exception):
    """
    The requested dataset does not exist, or is not implemented.
    """
    pass


class SgLibraryDatasets(object):
    """
    Holds all of the different library dataset dictionaries, by DL Task mapping

        Attributes:
            CLASSIFICATION          Dictionary of Classification Data sets
            OBJECT_DETECTION        Dictionary of Object Detection Data sets
            SEMANTIC_SEGMENTATION   Dictionary of Semantic Segmentation Data sets
    """
    CLASSIFICATION = CLASSIFICATION_DATASETS
    OBJECT_DETECTION = OBJECT_DETECTION_DATASETS
    SEMANTIC_SEGMENTATION = SEMANTIC_SEGMENTATION_DATASETS

    _datasets_mapping = {
        DeepLearningTask.CLASSIFICATION: CLASSIFICATION,
        DeepLearningTask.SEMANTIC_SEGMENTATION: SEMANTIC_SEGMENTATION,
        DeepLearningTask.OBJECT_DETECTION: OBJECT_DETECTION,
    }

    @staticmethod
    def get_all_available_datasets() -> Dict[str, List[str]]:
        """
        Gets all the available datasets.
        """
        all_datasets: Dict[str, List[str]] = defaultdict(list)
        for dl_task, task_datasets in SgLibraryDatasets._datasets_mapping.items():
            for dataset_name, dataset_interface in task_datasets.items():
                all_datasets[dl_task].append(dataset_name)

        # TODO: Return Dataset Metadata list from the dataset interfaces objects
        # TODO: Transform DatasetInterface -> DataSetMetadata
        return all_datasets

    @staticmethod
    def get_dataset(dl_task: str, dataset_name: str) -> Type[DatasetInterface]:
        """
        Get's a dataset with a given name for a given deep learning task.
        examp:
        >>> SgLibraryDatasets.get_dataset(dl_task='classification', dataset_name='cifar_100')
        >>> <Cifar100DatasetInterface instance>
        """
        task_datasets: Dict[str, DatasetInterface] = SgLibraryDatasets._datasets_mapping.get(dl_task)
        if not task_datasets:
            raise ValueError(f"Invalid Deep Learining Task: {dl_task}")

        dataset: DatasetInterface = task_datasets.get(dataset_name)
        if not dataset:
            raise DataSetDoesNotExistException(dataset_name)

        return dataset
