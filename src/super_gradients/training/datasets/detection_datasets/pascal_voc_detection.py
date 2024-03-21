import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
from xml.etree import ElementTree

from torch.utils.data import ConcatDataset
from tqdm import tqdm

import numpy as np

from super_gradients.common.deprecate import deprecated_parameter
from super_gradients.common.object_names import Datasets
from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.datasets.detection_datasets.pascal_voc_format_detection import PascalVOCFormatDetectionDataset
from super_gradients.training.transforms.transforms import AbstractDetectionTransform
from super_gradients.training.utils.utils import download_and_untar_from_url
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.datasets_conf import PASCAL_VOC_2012_CLASSES_LIST

logger = get_logger(__name__)


@register_dataset(Datasets.PASCAL_VOC_DETECTION_DATASET)
class PascalVOCDetectionDataset(PascalVOCFormatDetectionDataset):
    """Dataset for Pascal VOC object detection

        Parameters:
            data_dir (str): Base directory where the dataset is stored.
            images_dir (str, optional): Directory containing all the images, relative to `data_dir`. Defaults to None.
            labels_dir (str, optional): Directory containing all the labels, relative to `data_dir`. Defaults to None.
            images_sub_directory (str, optional): Deprecated. Subdirectory within data_dir that includes images. Defaults to None.
            download (bool, optional): If True, download the dataset to `data_dir`. Defaults to False.

        Dataset structure:

        ./data/pascal_voc
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

    Note:
        If both 'images_sub_directory' and ('images_dir', 'labels_dir') are provided, a warning will be raised.

    Usage:
        voc_2012_train = PascalVOCDetectionDataset(data_dir="./data/pascal_voc",
                                            images_dir="images/train2012/JPEGImages",
                                            labels_dir="labels/train2012/Annotations",
                                            download=True)
    """

    @deprecated_parameter(
        "images_sub_directory",
        deprecated_since="3.7.0",
        removed_from="3.8.0",
        reason="Support of `images_sub_directory` is removed since it allows less flexibility." " Please use 'images_dir' and 'labels_dir' instead.",
    )
    def __init__(
        self,
        data_dir: str,
        images_sub_directory: Optional[str] = None,
        images_dir: Optional[str] = None,
        labels_dir: Optional[str] = None,
        download: bool = False,
        max_num_samples: int = None,
        cache_annotations: bool = True,
        input_dim: Union[int, Tuple[int, int], None] = None,
        transforms: List[AbstractDetectionTransform] = [],
        class_inclusion_list: Optional[List[str]] = None,
        ignore_empty_annotations: bool = True,
        verbose: bool = True,
        show_all_warnings: bool = False,
        cache=None,
        cache_dir=None,
    ):
        """
        Initialize the Pascal VOC Detection Dataset.

        """

        # Adding a check for deprecated usage alongside new parameters
        if images_sub_directory is not None and (images_dir is not None or labels_dir is not None):
            logger.warning(
                "Both 'images_sub_directory' (deprecated) and 'images_dir'/'labels_dir' are provided. "
                "Prefer using 'images_dir' and 'labels_dir' for future compatibility.",
                DeprecationWarning,
            )

        elif images_sub_directory is not None:
            images_dir = images_sub_directory
            labels_dir = images_sub_directory.replace("images", "labels")
        elif images_dir is None or labels_dir is None:
            raise ValueError("You must provide either 'images_dir' and 'labels_dir', or the deprecated 'images_sub_directory'.")

        if download:
            self.download(data_dir)

        super().__init__(
            data_dir=data_dir,
            images_dir=images_dir,
            labels_dir=labels_dir,
            max_num_samples=max_num_samples,
            cache_annotations=cache_annotations,
            input_dim=input_dim,
            transforms=transforms,
            class_inclusion_list=class_inclusion_list,
            ignore_empty_annotations=ignore_empty_annotations,
            verbose=verbose,
            show_all_warnings=show_all_warnings,
            cache=cache,
            cache_dir=cache_dir,
            all_classes_list=PASCAL_VOC_2012_CLASSES_LIST,
            label_file_ext="txt",
        )

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
    """Unified Dataset for Pascal VOC object detection.

    Unified Dataset class for training on Pascal VOC object detection datasets.

    This class combines datasets from multiple years (e.g., 2007, 2012) into a single dataset for training purposes.

    Parameters:
        data_dir (str): Base directory where the dataset is stored.
        input_dim (tuple): Input dimension that the images should be resized to.
        cache (optional): Cache configuration.
        cache_dir (optional): Directory for cache.
        transforms (List[AbstractDetectionTransform], optional): List of transforms to apply.
        class_inclusion_list (Optional[List[str]], optional): List of classes to include.
        max_num_samples (int, optional): Maximum number of samples to include from each dataset part.
        download (bool, optional): If True, downloads the dataset parts to `data_dir`. Defaults to False.
        images_dir (Optional[str], optional): Directory containing all the images, relative to `data_dir`. Should only be used without 'images_sub_directory'.
        labels_dir (Optional[str], optional): Directory containing all the labels, relative to `data_dir`. Should only be used without 'images_sub_directory'.
        images_sub_directory (Optional[str], optional): Deprecated. Use 'images_dir' and 'labels_dir' instead for future compatibility.


        Example Dataset structure:

            ./data/pascal_voc/
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
            Usage:
        unified_dataset = PascalVOCUnifiedDetectionTrainDataset(data_dir="./data/pascal_voc",
                                                                input_dim=(512, 512),
                                                                download=True,
                                                                images_dir="images",
                                                                labels_dir="labels")

    """

    @deprecated_parameter(
        "images_sub_directory",
        deprecated_since="3.7.0",
        removed_from="3.8.0",
        reason="Support of `images_sub_directory` is removed since it allows less flexibility. Please use " "'images_dir' and 'labels_dir' instead.",
    )
    def __init__(
        self,
        data_dir: str,
        input_dim: tuple,
        cache=None,
        cache_dir=None,
        transforms: List[AbstractDetectionTransform] = [],
        class_inclusion_list: Optional[List[str]] = None,
        max_num_samples: int = None,
        download: bool = False,
        images_dir: Optional[str] = None,
        labels_dir: Optional[str] = None,
        images_sub_directory: Optional[str] = None,  # Marked for deprecation.
    ):
        if images_sub_directory is not None and (images_dir is not None or labels_dir is not None):
            logger.warning(
                "Both 'images_sub_directory' (deprecated) and 'images_dir'/'labels_dir' are provided. "
                "Prefer using 'images_dir' and 'labels_dir' for future compatibility.",
                DeprecationWarning,
            )
        if download:
            PascalVOCDetectionDataset.download(data_dir=data_dir)

        train_dataset_names = ["train2007", "val2007", "train2012", "val2012"]
        if max_num_samples:
            max_num_samples_per_train_dataset = [len(segment) for segment in np.array_split(range(max_num_samples), len(train_dataset_names))]
        else:
            max_num_samples_per_train_dataset = [None] * len(train_dataset_names)

        train_sets = []
        for i, trainset_name in enumerate(train_dataset_names):
            dataset_kwargs = {
                "data_dir": data_dir,
                "input_dim": input_dim,
                "cache": cache,
                "cache_dir": cache_dir,
                "transforms": transforms,
                "class_inclusion_list": class_inclusion_list,
                "max_num_samples": max_num_samples_per_train_dataset[i],
            }
            if images_dir is not None and labels_dir is not None:
                dataset_kwargs["images_dir"] = os.path.join(images_dir, trainset_name)
                dataset_kwargs["labels_dir"] = os.path.join(labels_dir, trainset_name)
            elif images_sub_directory is not None:
                deprecated_images_path = os.path.join("images", trainset_name)
                deprecated_labels_path = os.path.join("labels", trainset_name)
                dataset_kwargs["images_dir"] = deprecated_images_path
                dataset_kwargs["labels_dir"] = deprecated_labels_path
            else:
                raise ValueError("You must provide either 'images_dir' and 'labels_dir', or the deprecated 'images_sub_directory'.")

            train_sets.append(PascalVOCDetectionDataset(**dataset_kwargs))
            super(PascalVOCUnifiedDetectionTrainDataset, self).__init__(train_sets)
