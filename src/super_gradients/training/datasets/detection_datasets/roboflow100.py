import os

from tqdm import tqdm
from torch.utils.data import ConcatDataset

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormatDetectionDataset

logger = get_logger(__name__)
ALL_ROBOFLOW_DATASETS = [
    "4-fold-defect",
    "abdomen-mri",
    "acl-x-ray",
    "activity-diagrams-qdobr",
    "aerial-cows",
    "aerial-pool",
    "aerial-spheres",
    "animals-ij5d2",
    "apex-videogame",
    "apples-fvpl5",
    "aquarium-qlnqy",
    "asbestos",
    "avatar-recognition-nuexe",
    "axial-mri",
    "bacteria-ptywi",
    "bccd-ouzjz",
    "bees-jt5in",
    "bone-fracture-7fylg",
    "brain-tumor-m2pbp",
    "cable-damage",
    "cables-nl42k",
    "cavity-rs0uf",
    "cell-towers",
    "cells-uyemf",
    "chess-pieces-mjzgj",
    "circuit-elements",
    "circuit-voltages",
    "cloud-types",
    "coins-1apki",
    "construction-safety-gsnvb",
    "coral-lwptl",
    "corrosion-bi3q3",
    "cotton-20xz5",
    "cotton-plant-disease",
    "csgo-videogame",
    "currency-v4f8j",
    "digits-t2eg6",
    "document-parts",
    "excavators-czvg9",
    "farcry6-videogame",
    "fish-market-ggjso",
    "flir-camera-objects",
    "furniture-ngpea",
    "gauge-u2lwv",
    "grass-weeds",
    "gynecology-mri",
    "halo-infinite-angel-videogame",
    "hand-gestures-jps7z",
    "insects-mytwu",
    "leaf-disease-nsdsr",
    "lettuce-pallets",
    "liver-disease",
    "marbles",
    "mask-wearing-608pr",
    "mitosis-gjs3g",
    "number-ops",
    "paper-parts",
    "paragraphs-co84b",
    "parasites-1s07h",
    "peanuts-sd4kf",
    "peixos-fish",
    "people-in-paintings",
    "pests-2xlvx",
    "phages",
    "pills-sxdht",
    "poker-cards-cxcvz",
    "printed-circuit-board",
    "radio-signal",
    "road-signs-6ih4y",
    "road-traffic",
    "robomasters-285km",
    "secondary-chains",
    "sedimentary-features-9eosf",
    "shark-teeth-5atku",
    "sign-language-sokdr",
    "signatures-xc8up",
    "smoke-uvylj",
    "soccer-players-5fuqs",
    "soda-bottles",
    "solar-panels-taxvb",
    "stomata-cells",
    "street-work",
    "tabular-data-wf9uh",
    "team-fight-tactics",
    "thermal-cheetah-my4dp",
    "thermal-dogs-and-people-x6ejw",
    "trail-camera",
    "truck-movement",
    "tweeter-posts",
    "tweeter-profile",
    "underwater-objects-5v7p8",
    "underwater-pipes-4ng4t",
    "uno-deck",
    "valentines-chocolate",
    "vehicles-q0x2v",
    "wall-damage",
    "washroom-rf1fa",
    "weed-crop-aerial",
    "wine-labels",
    "x-ray-rheumatology",
]


class RoboflowDetectionDataset(COCOFormatDetectionDataset):
    """Dataset for Roboflow100 object detection. TODO: improve

    To use this Dataset you need to:

        - Follow the official instructions to download Roboflow100: https://github.com/roboflow/roboflow-100-benchmark?ref=roboflow-blog

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

        - Instantiate the dataset:
            >> train_set = RoboflowDetectionDataset(data_dir='.../rf100', dataset_name="digits-t2eg6", split="train")
            >> valid_set = RoboflowDetectionDataset(data_dir='.../rf100', dataset_name="digits-t2eg6", split="valid")
    """

    def __init__(self, data_dir: str, dataset_name: str, split: str, *args, **kwargs):
        """
        :param data_dir:        Where the data is stored.
        :param dataset_name:    One of the 100 dataset name. (You can run RoboflowDetectionDataset.list_roboflow_datasets() to see all available datasets)
        :param split:           train, valid or test.
        """
        if split not in ("train", "valid", "test"):
            raise ValueError(f"split must be one of ('train', 'valid', 'test'). Got '{split}'.")

        dataset_split_dir = os.path.join(dataset_name, split)
        json_annotation_file = os.path.join(dataset_split_dir, "_annotations.coco.json")
        super().__init__(data_dir=data_dir, json_annotation_file=json_annotation_file, images_dir=dataset_split_dir, *args, **kwargs)

    @staticmethod
    def list_roboflow_datasets() -> list:
        return ALL_ROBOFLOW_DATASETS


class Roboflow100DetectionDataset(ConcatDataset):
    """Dataset for Roboflow100 object detection. TODO: improve

    To use this Dataset you need to:

        - Follow the official instructions to download Roboflow100: https://github.com/roboflow/roboflow-100-benchmark?ref=roboflow-blog

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

        - Instantiate the dataset:
            >> train_set = RoboflowDetectionDataset(data_dir='.../rf100', dataset_name="digits-t2eg6", split="train")
            >> valid_set = RoboflowDetectionDataset(data_dir='.../rf100', dataset_name="digits-t2eg6", split="valid")
    """

    def __init__(self, data_dir: str, split: str, verbose: bool = True, *args, **kwargs):
        """
        :param data_dir:        Where the data is stored.
        :param split:           train, valid or test.
        """

        if "max_num_samples" in kwargs:
            logger.warning("max_num_samples will be ignored because it is not supported by Roboflow100DetectionDataset.")
            kwargs["max_num_samples"] = None

        from contextlib import redirect_stdout

        with redirect_stdout(open(os.devnull, "w")):  # Fixme: remove this, but instead hide coco prints
            roboflow_datasets = [
                RoboflowDetectionDataset(data_dir=data_dir, dataset_name=dataset_name, split=split, verbose=False, *args, **kwargs)
                for dataset_name in tqdm(RoboflowDetectionDataset.list_roboflow_datasets()[:2], disable=not verbose)
            ]
        super(Roboflow100DetectionDataset, self).__init__(roboflow_datasets)


# data = Roboflow100DetectionDataset(data_dir="/data/rf100", split="train", input_dim=(640, 640))


# data.plot()
# print(1)
