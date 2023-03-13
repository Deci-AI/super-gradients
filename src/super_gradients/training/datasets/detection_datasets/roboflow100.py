import os
from typing import List

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormattedDetectionDataset

logger = get_logger(__name__)


DATASET_CATEGORIES = (
    "real world",
    "videogames",
    "documents",
    "underwater",
    "aerial",
    "microscopic",
    "electromagnetic",
)

ROBOFLOW_DATASETS_NAMES_WITH_CATEGORY = {
    "hand-gestures-jps7z": "real world",
    "smoke-uvylj": "real world",
    "wall-damage": "real world",
    "corrosion-bi3q3": "real world",
    "excavators-czvg9": "real world",
    "chess-pieces-mjzgj": "real world",
    "road-signs-6ih4y": "real world",
    "street-work": "real world",
    "construction-safety-gsnvb": "real world",
    "road-traffic": "real world",
    "washroom-rf1fa": "real world",
    "circuit-elements": "real world",
    "mask-wearing-608pr": "real world",
    "cables-nl42k": "real world",
    "soda-bottles": "real world",
    "truck-movement": "real world",
    "wine-labels": "real world",
    "digits-t2eg6": "real world",
    "vehicles-q0x2v": "real world",
    "peanuts-sd4kf": "real world",
    "printed-circuit-board": "real world",
    "pests-2xlvx": "real world",
    "cavity-rs0uf": "real world",
    "leaf-disease-nsdsr": "real world",
    "marbles": "real world",
    "pills-sxdht": "real world",
    "poker-cards-cxcvz": "real world",
    "number-ops": "real world",
    "insects-mytwu": "real world",
    "cotton-20xz5": "real world",
    "furniture-ngpea": "real world",
    "cable-damage": "real world",
    "animals-ij5d2": "real world",
    "coins-1apki": "real world",
    "apples-fvpl5": "real world",
    "people-in-paintings": "real world",
    "circuit-voltages": "real world",
    "uno-deck": "real world",
    "grass-weeds": "real world",
    "gauge-u2lwv": "real world",
    "sign-language-sokdr": "real world",
    "valentines-chocolate": "real world",
    "fish-market-ggjso": "real world",
    "lettuce-pallets": "real world",
    "shark-teeth-5atku": "real world",
    "bees-jt5in": "real world",
    "sedimentary-features-9eosf": "real world",
    "currency-v4f8j": "real world",
    "trail-camera": "real world",
    "cell-towers": "real world",
    "apex-videogame": "videogames",
    "farcry6-videogame": "videogames",
    "csgo-videogame": "videogames",
    "avatar-recognition-nuexe": "videogames",
    "halo-infinite-angel-videogame": "videogames",
    "team-fight-tactics": "videogames",
    "robomasters-285km": "videogames",
    "tweeter-posts": "documents",
    "tweeter-profile": "documents",
    "document-parts": "documents",
    "activity-diagrams-qdobr": "documents",
    "signatures-xc8up": "documents",
    "paper-parts": "documents",
    "tabular-data-wf9uh": "documents",
    "paragraphs-co84b": "documents",
    "underwater-pipes-4ng4t": "underwater",
    "aquarium-qlnqy": "underwater",
    "peixos-fish": "underwater",
    "underwater-objects-5v7p8": "underwater",
    "coral-lwptl": "underwater",
    "aerial-pool": "aerial",
    "secondary-chains": "aerial",
    "aerial-spheres": "aerial",
    "soccer-players-5fuqs": "aerial",
    "weed-crop-aerial": "aerial",
    "aerial-cows": "aerial",
    "cloud-types": "aerial",
    "stomata-cells": "microscopic",
    "bccd-ouzjz": "microscopic",
    "parasites-1s07h": "microscopic",
    "cells-uyemf": "microscopic",
    "4-fold-defect": "microscopic",
    "bacteria-ptywi": "microscopic",
    "cotton-plant-disease": "microscopic",
    "mitosis-gjs3g": "microscopic",
    "phages": "microscopic",
    "liver-disease": "microscopic",
    "asbestos": "microscopic",
    "thermal-dogs-and-people-x6ejw": "electromagnetic",
    "solar-panels-taxvb": "electromagnetic",
    "radio-signal": "electromagnetic",
    "thermal-cheetah-my4dp": "electromagnetic",
    "x-ray-rheumatology": "electromagnetic",
    "acl-x-ray": "electromagnetic",
    "abdomen-mri": "electromagnetic",
    "axial-mri": "electromagnetic",
    "gynecology-mri": "electromagnetic",
    "brain-tumor-m2pbp": "electromagnetic",
    "bone-fracture-7fylg": "electromagnetic",
    "flir-camera-objects": "electromagnetic",
}


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
        :param dataset_name:    One of the 100 dataset name. (You can run RoboflowDetectionDataset.list_roboflow_datasets() to see all available datasets)
        :param split:           train, valid or test.
        """
        if split not in ("train", "valid", "test"):
            raise ValueError(f"split must be one of ('train', 'valid', 'test'). Got '{split}'.")

        self.dataset_name = dataset_name
        dataset_split_dir = os.path.join(dataset_name, split)
        json_annotation_file = os.path.join(dataset_split_dir, "_annotations.coco.json")

        super().__init__(data_dir=data_dir, json_annotation_file=json_annotation_file, images_dir=dataset_split_dir, *args, **kwargs)

    @staticmethod
    def list_roboflow_datasets(categories: List[str] = DATASET_CATEGORIES) -> List[str]:
        """List all available datasets of specified categories. By default, select all the datasets."""
        return [dataset_name for dataset_name, category in ROBOFLOW_DATASETS_NAMES_WITH_CATEGORY.items() if category in categories]

    @property
    def category(self) -> str:
        """Category of the dataset."""
        _category = ROBOFLOW_DATASETS_NAMES_WITH_CATEGORY.get(self.dataset_name)
        if _category is None:
            logger.warning(f"No category found for dataset_name={self.dataset_name}. This might be due to a recent change in the dataset name.")
        return _category
