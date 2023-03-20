import os
from typing import List, Dict, Union, Optional

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormattedDetectionDataset
from super_gradients.training.datasets.detection_datasets.roboflow.utils import get_dataset_metadata, list_datasets

logger = get_logger(__name__)


class RoboflowDetectionDataset(COCOFormattedDetectionDataset):
    """Dataset that can be used with ANY of the Roboflow100 benchmark datasets for object detection.

    To use this Dataset you need to:

        - Follow the official instructions to download Roboflow100: https://github.com/roboflow/roboflow-100-benchmark?ref=roboflow-blog
            //!\\ To use this dataset, you have to download the "coco" format, NOT the yolov5.

        - Your dataset should loook like this:
            rf100
            ├── 4-fold-defect
            │      ├─ train
            │      │    ├─ 000000000001.jpg
            │      │    ├─ ...
            │      │    └─ _annotations.coco.json
            │      ├─ valid
            │      │    └─ ...
            │      └─ test
            │           └─ ...
            ├── abdomen-mri
            │      └─ ...
            └── ...

        - Install CoCo API: https://github.com/pdollar/coco/tree/master/PythonAPI

        - Instantiate the dataset (in this case we load the dataset called "digits-t2eg6")"
            >> train_set = RoboflowDetectionDataset(data_dir='<path-to>/rf100', dataset_name="digits-t2eg6", split="train")
            >> valid_set = RoboflowDetectionDataset(data_dir='<path-to>/rf100', dataset_name="digits-t2eg6", split="valid")
    """

    def __init__(self, data_dir: str, dataset_name: str, split: str, *args, **kwargs):
        """
        :param data_dir:        Where the data is stored.
        :param dataset_name:    One of the 100 dataset name. (You can run RoboflowDetectionDataset.list_datasets() to see all available datasets)
        :param split:           train, valid or test.
        """
        if split not in ("train", "valid", "test"):
            raise ValueError(f"split must be one of ('train', 'valid', 'test'). Got '{split}'.")

        self.dataset_name = dataset_name
        dataset_split_dir = os.path.join(dataset_name, split)
        json_annotation_file = os.path.join(dataset_split_dir, "_annotations.coco.json")

        super().__init__(data_dir=data_dir, json_annotation_file=json_annotation_file, images_dir=dataset_split_dir, *args, **kwargs)

    @staticmethod
    def list_datasets(categories: Optional[List[str]] = None) -> List[str]:
        """List all available datasets of specified categories. By default, list all the datasets."""
        return list_datasets(categories=categories)

    @property
    def metadata(self) -> Optional[Dict[str, Union[str, int]]]:
        """Category of the dataset. Note that each dataset has one and only one category."""
        return get_dataset_metadata(self.dataset_name)
