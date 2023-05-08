import os
import glob
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree

from torch.utils.data import ConcatDataset
from tqdm import tqdm

import numpy as np

from super_gradients.common.object_names import Datasets
from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.transforms.transforms import DetectionTransform
from super_gradients.training.utils.utils import download_and_untar_from_url, get_image_size_from_path
from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataset
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.datasets_conf import PASCAL_VOC_2012_CLASSES_LIST

logger = get_logger(__name__)


@register_dataset(Datasets.PASCAL_VOC_DETECTION_DATASET)
class PascalVOCDetectionDataset(DetectionDataset):
    """Dataset for Pascal VOC object detection

    To use this Dataset you need to:
        >> train_set = PascalVOCDetectionDataset(download=True, ...)

    Dataset structure:
        ├─images
        │   ├─ train2012
        │   ├─ val2012
        │   ├─ VOCdevkit
        │   │    ├─ VOC2007
        │   │    │  ├──JPEGImages
        │   │    │  ├──SegmentationClass
        │   │    │  ├──ImageSets
        │   │    │  ├──ImageSets/Segmentation
        │   │    │  ├──ImageSets/Main
        │   │    │  ├──ImageSets/Layout
        │   │    │  ├──Annotations
        │   │    │  └──SegmentationObject
        │   │    └──VOC2012
        │   │       ├──JPEGImages
        │   │       ├──SegmentationClass
        │   │       ├──ImageSets
        │   │       ├──ImageSets/Segmentation
        │   │       ├──ImageSets/Main
        │   │       ├──ImageSets/Action
        │   │       ├──ImageSets/Layout
        │   │       ├──Annotations
        │   │       └──SegmentationObject
        │   ├─train2007
        │   ├─test2007
        │   └─val2007
        └─labels
            ├─train2012
            ├─val2012
            ├─train2007
            ├─test2007
            └─val2007

    """

    def __init__(self, images_sub_directory: str, download: bool = False, *args, **kwargs):
        """Dataset for Pascal VOC object detection

        :param images_sub_directory:    Sub directory of data_dir that includes images.
        """

        self.images_sub_directory = images_sub_directory
        self.img_and_target_path_list = None
        data_dir = kwargs.get("data_dir")
        if data_dir is None:
            raise ValueError("Must pass data_dir != None through **kwargs")
        if download:
            PascalVOCDetectionDataset.download(data_dir)

        kwargs["original_target_format"] = DetectionTargetsFormat.XYXY_LABEL
        kwargs["all_classes_list"] = PASCAL_VOC_2012_CLASSES_LIST
        super().__init__(*args, **kwargs)

    def _setup_data_source(self) -> int:
        """Initialize img_and_target_path_list and warn if label file is missing

        :return: List of tuples made of (img_path,target_path)
        """
        img_files_folder = os.path.join(self.data_dir, self.images_sub_directory)
        if not Path(img_files_folder).exists():
            raise FileNotFoundError(
                f"{img_files_folder} not found...\n"
                f"Please make sure that f{self.data_dir} points toward your PascalVOC dataset folder.\n"
                f"If you don't have it locally, you can set PascalVOCDetectionDataset(..., download=True)"
            )

        img_files = glob.glob(img_files_folder + "*.jpg")
        if len(img_files) == 0:
            raise FileNotFoundError(f"No image file found at {img_files_folder}")

        target_files = [img_file.replace("images", "labels").replace(".jpg", ".txt") for img_file in img_files]

        img_and_target_path_list = [(img_file, target_file) for img_file, target_file in zip(img_files, target_files) if os.path.exists(target_file)]
        if len(img_and_target_path_list) == 0:
            raise FileNotFoundError("No target file associated to the images was found")

        num_missing_files = len(img_files) - len(img_and_target_path_list)
        if num_missing_files > 0:
            logger.warning(f"{num_missing_files} labels files were not loaded our of {len(img_files)} image files")

        self.img_and_target_path_list = img_and_target_path_list
        return len(self.img_and_target_path_list)

    def _load_annotation(self, sample_id: int) -> dict:
        """Load annotations associated to a specific sample.

        :return: Annotation including:
                    - target in XYXY_LABEL format
                    - img_path
        """
        img_path, target_path = self.img_and_target_path_list[sample_id]
        with open(target_path, "r") as targets_file:
            target = np.array([x.split() for x in targets_file.read().splitlines()], dtype=np.float32)

        height, width = get_image_size_from_path(img_path)

        # We have to rescale the targets because the images will be resized.
        r = min(self.input_dim[1] / height, self.input_dim[0] / width)
        target[:, :4] *= r

        resized_img_shape = (int(height * r), int(width * r))

        return {"img_path": img_path, "target": target, "resized_img_shape": resized_img_shape}

    @staticmethod
    def download(data_dir: str) -> None:
        """Download Pascal dataset in XYXY_LABEL format.

        Data extracted form http://host.robots.ox.ac.uk/pascal/VOC/
        """

        def _parse_and_save_labels(path: str, new_label_path: str, year: str, image_id: str) -> None:
            """Parse and save the labels of an image in XYXY_LABEL format."""

            with open(f"{path}/VOC{year}/Annotations/{image_id}.xml") as f:
                xml_parser = ElementTree.parse(f).getroot()

            labels = []
            for obj in xml_parser.iter("object"):
                cls = obj.find("name").text
                if cls in PASCAL_VOC_2012_CLASSES_LIST and not int(obj.find("difficult").text) == 1:
                    xml_box = obj.find("bndbox")

                    def get_coord(box_coord):
                        return xml_box.find(box_coord).text

                    xmin, ymin, xmax, ymax = get_coord("xmin"), get_coord("ymin"), get_coord("xmax"), get_coord("ymax")
                    labels.append(" ".join([xmin, ymin, xmax, ymax, str(PASCAL_VOC_2012_CLASSES_LIST.index(cls))]))

            with open(new_label_path, "w") as f:
                f.write("\n".join(labels))

        urls = [
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",  # 439M 5011 images
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",  # 430M, 4952 images
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        ]  # 1.86G, 17125 images
        data_dir = Path(data_dir)
        download_and_untar_from_url(urls, dir=data_dir / "images")

        # Convert
        data_path = data_dir / "images" / "VOCdevkit"
        for year, image_set in ("2012", "train"), ("2012", "val"), ("2007", "train"), ("2007", "val"), ("2007", "test"):
            dest_imgs_path = data_dir / "images" / f"{image_set}{year}"
            dest_imgs_path.mkdir(exist_ok=True, parents=True)

            dest_labels_path = data_dir / "labels" / f"{image_set}{year}"
            dest_labels_path.mkdir(exist_ok=True, parents=True)

            with open(data_path / f"VOC{year}/ImageSets/Main/{image_set}.txt") as f:
                image_ids = f.read().strip().split()

            for id in tqdm(image_ids, desc=f"{image_set}{year}"):
                img_path = data_path / f"VOC{year}/JPEGImages/{id}.jpg"
                new_img_path = dest_imgs_path / img_path.name
                new_label_path = (dest_labels_path / img_path.name).with_suffix(".txt")
                img_path.rename(new_img_path)  # Move image to dest folder
                _parse_and_save_labels(data_path, new_label_path, year, id)


class PascalVOCUnifiedDetectionTrainDataset(ConcatDataset):
    """Unified Dataset for Pascal VOC object detection

    To use this Dataset you need to:
        >> train_set = PascalVOCUnifiedDetectionTrainDataset(download=True, ...)

    Dataset structure:
        ├─images
        │   ├─ train2012
        │   ├─ val2012
        │   ├─ VOCdevkit
        │   │    ├─ VOC2007
        │   │    │  ├──JPEGImages
        │   │    │  ├──SegmentationClass
        │   │    │  ├──ImageSets
        │   │    │  ├──ImageSets/Segmentation
        │   │    │  ├──ImageSets/Main
        │   │    │  ├──ImageSets/Layout
        │   │    │  ├──Annotations
        │   │    │  └──SegmentationObject
        │   │    └──VOC2012
        │   │       ├──JPEGImages
        │   │       ├──SegmentationClass
        │   │       ├──ImageSets
        │   │       ├──ImageSets/Segmentation
        │   │       ├──ImageSets/Main
        │   │       ├──ImageSets/Action
        │   │       ├──ImageSets/Layout
        │   │       ├──Annotations
        │   │       └──SegmentationObject
        │   ├─train2007
        │   ├─test2007
        │   └─val2007
        └─labels
            ├─train2012
            ├─val2012
            ├─train2007
            ├─test2007
            └─val2007
    """

    def __init__(
        self,
        data_dir: str,
        input_dim: tuple,
        cache: bool = False,
        cache_dir: str = None,
        transforms: List[DetectionTransform] = [],
        class_inclusion_list: Optional[List[str]] = None,
        max_num_samples: int = None,
        download: bool = False,
    ):
        if download:
            PascalVOCDetectionDataset.download(data_dir=data_dir)

        train_dataset_names = ["train2007", "val2007", "train2012", "val2012"]
        # We divide train_max_num_samples between the datasets
        if max_num_samples:
            max_num_samples_per_train_dataset = [len(segment) for segment in np.array_split(range(max_num_samples), len(train_dataset_names))]
        else:
            max_num_samples_per_train_dataset = [None] * len(train_dataset_names)
        train_sets = [
            PascalVOCDetectionDataset(
                data_dir=data_dir,
                input_dim=input_dim,
                cache=cache,
                cache_dir=cache_dir,
                transforms=transforms,
                images_sub_directory="images/" + trainset_name + "/",
                class_inclusion_list=class_inclusion_list,
                max_num_samples=max_num_samples_per_train_dataset[i],
            )
            for i, trainset_name in enumerate(train_dataset_names)
        ]
        super(PascalVOCUnifiedDetectionTrainDataset, self).__init__(train_sets)
