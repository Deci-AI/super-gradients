from typing import Dict, Union, Optional, List

from super_gradients.training.datasets.detection_datasets.roboflow.metadata import DATASETS_METADATA, DATASETS_CATEGORIES
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


def list_datasets(categories: List[str] = None) -> List[str]:
    """List all available datasets of specified categories. By default, list all the datasets."""
    categories = categories or DATASETS_CATEGORIES
    return [dataset_name for dataset_name, metadata in DATASETS_METADATA.items() if metadata["category"] in categories]


def get_dataset_metadata(dataset_name: str) -> Optional[Dict[str, Union[str, int]]]:
    """Get the metadata of a specific roboflow dataset.
    :param dataset_name: Name of the dataset, as listed in the official repo -
                            https://github.com/roboflow/roboflow-100-benchmark/blob/main/metadata/datasets_stats.csv
    :return:             Metadata of the dataset
    """
    dataset_metadata = DATASETS_METADATA.get(dataset_name)
    if dataset_metadata is None:
        logger.warning(f"No metadata found for dataset_name={dataset_name}. This might be due to a recent change in the dataset name.")
    return dataset_metadata


def get_dataset_num_classes(dataset_name: str) -> int:
    """Get the number of classes of a specific roboflow dataset.
    :param dataset_name: Name of the dataset, as listed in the official repo -
                            https://github.com/roboflow/roboflow-100-benchmark/blob/main/metadata/datasets_stats.csv
    :return:             Number of classes of the dataset. Note that the number of classes in the official documentation is different to the actual one.
    """
    metadata = get_dataset_metadata(dataset_name)
    if metadata is None:
        raise ValueError(f"No num_classes found for dataset_name={dataset_name}. This might be due to a recent change in the dataset name.")
    return metadata["num_classes_found"]
