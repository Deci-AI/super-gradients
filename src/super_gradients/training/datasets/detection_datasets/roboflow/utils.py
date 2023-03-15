from typing import Dict, Union, Optional

from super_gradients.training.datasets.detection_datasets.roboflow.metadata import DATASETS_METADATA
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


def get_dataset_metadata(dataset_name: str) -> Optional[Dict[str, Union[str, int]]]:
    dataset_metadata = DATASETS_METADATA.get(dataset_name)
    if dataset_metadata is None:
        logger.warning(f"No metadata found for dataset_name={dataset_name}. This might be due to a recent change in the dataset name.")
    return dataset_metadata


def get_dataset_num_classes(dataset_name: str) -> int:
    metadata = get_dataset_metadata(dataset_name)
    if metadata is None:
        raise ValueError(f"No num_classes found for dataset_name={dataset_name}. This might be due to a recent change in the dataset name.")
    return metadata["num_classes"]
