import json
from typing import Dict, Union, Tuple, Optional
from pathlib import Path

from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


DATASET_METADATA_FILENAME = "datasets_metadata.json"
DATASET_METADATA_FIELDS = ["category", "train", "test", "valid", "size", "num_classes"]


def fetch_datasets_metadata():
    """Fetch the dataset statistics from the official roboflow repository. Convert it from csv to json, and save it locally."""

    import pandas as pd

    # Raw content of: https://github.com/roboflow/roboflow-100-benchmark/blob/main/metadata/datasets_stats.csv
    df = pd.read_csv("https://raw.githubusercontent.com/roboflow/roboflow-100-benchmark/main/metadata/datasets_stats.csv")

    # Select only relevant columns
    df = df[["dataset"] + DATASET_METADATA_FIELDS]

    df["num_classes"] = df["num_classes"].astype(int)
    df.set_index("dataset").to_json(DATASET_METADATA_FILENAME, orient="index")


def get_datasets_metadata() -> Dict[str, Dict[str, Union[str, int]]]:
    local_dir = Path(__file__).parent
    with open(local_dir / DATASET_METADATA_FILENAME, "r") as f:
        return json.load(f)


def get_categories() -> Tuple:
    dataset_metadata = get_datasets_metadata()
    return tuple(set(metadata["category"] for metadata in dataset_metadata.values()))


def get_dataset_metadata(dataset_name: str) -> Optional[Dict[str, Union[str, int]]]:
    dataset_metadata = get_datasets_metadata()
    if dataset_metadata is None:
        logger.warning(f"No metadata found for dataset_name={dataset_name}. This might be due to a recent change in the dataset name.")
    return dataset_metadata


def get_dataset_num_classes(dataset_name: str) -> int:
    metadata = get_dataset_metadata(dataset_name)
    if metadata is None:
        raise ValueError(f"No num_classes found for dataset_name={dataset_name}. This might be due to a recent change in the dataset name.")
    return metadata["num_classes"]


DATASETS_METADATA = get_datasets_metadata()
DATASETS_CATEGORIES = get_categories()

if __name__ == "__main__":
    fetch_datasets_metadata()
