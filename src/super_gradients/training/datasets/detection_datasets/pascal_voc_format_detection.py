import os
import glob
from pathlib import Path
from typing import Optional, Union, List, Tuple

import numpy as np

from super_gradients.common.deprecate import deprecated_parameter
from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.transforms import AbstractDetectionTransform
from super_gradients.training.utils.utils import get_image_size_from_path
from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataset
from super_gradients.training.datasets.data_formats.default_formats import XYXY_LABEL
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


@register_dataset("PascalVOCFormatDetectionDataset")
class PascalVOCFormatDetectionDataset(DetectionDataset):
    """Dataset for Pascal VOC object detection

    Parameters:
        data_dir (str): Base directory where the dataset is stored.

        images_dir (Optional[str]): Directory containing all the images, relative to `data_dir`. Defaults to None.

        labels_dir (Optional[str]): Directory containing all the labels, relative to `data_dir`. Defaults to None.

        max_num_samples (Optional[int]): If not None, sets the maximum size of the dataset by only indexing the first
         n annotations/images. Defaults to None.

        cache_annotations (bool): Whether to cache annotations. Reduces training time by pre-loading all annotations
         but requires more RAM. Defaults to True.

        input_dim (Optional[Union[int, Tuple[int, int]]]): Image size when loaded, before transforms. Can be None, a scalar,
         or a tuple (height, width). Defaults to None.

        transforms (List[AbstractDetectionTransform]): List of transforms to apply sequentially on each sample.
         Defaults to an empty list.

        all_classes_list (Optional[List[str]]): All class names in the dataset. Defaults to an empty list.

        class_inclusion_list (Optional[List[str]]): Subset of classes to include. Classes not in this list will be excluded.
         Adjust the number of model classes accordingly. Defaults to None.

        ignore_empty_annotations (bool): If True and class_inclusion_list is not None, images without any target will be
         ignored. Defaults to True.

        verbose (bool): If True, displays additional information (does not include warnings). Defaults to True.

        show_all_warnings (bool): If True, displays all warnings. Defaults to False.

        cache (Optional): Deprecated. This parameter is not used and setting it has no effect. Will be removed in a
         future version.

        cache_dir (Optional): Deprecated. This parameter is not used and setting it has no effect. Will be removed in
         a future version.



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
        images_dir: str,
        labels_dir: str,
        max_num_samples: int = None,
        cache_annotations: bool = True,
        input_dim: Union[int, Tuple[int, int], None] = None,
        transforms: List[AbstractDetectionTransform] = [],
        all_classes_list: Optional[List[str]] = [],
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

        self.data_dir = data_dir

        self.images_dir = os.path.join(data_dir, images_dir)
        self.labels_dir = os.path.join(data_dir, labels_dir)

        super(PascalVOCFormatDetectionDataset, self).__init__(
            data_dir=data_dir,
            original_target_format=XYXY_LABEL,
            max_num_samples=max_num_samples,
            cache_annotations=cache_annotations,
            input_dim=input_dim,
            transforms=transforms,
            all_classes_list=all_classes_list,
            class_inclusion_list=class_inclusion_list,
            ignore_empty_annotations=ignore_empty_annotations,
            verbose=verbose,
            show_all_warnings=show_all_warnings,
            cache=cache,
            cache_dir=cache_dir,
        )

    def _setup_data_source(self) -> int:
        """Initialize img_and_target_path_list and warn if label file is missing

        :return: List of tuples made of (img_path,target_path)
        """
        if not Path(self.images_dir).exists():
            raise FileNotFoundError(f"{self.images_dir} not found.")

        img_files = list(sorted(glob.glob(os.path.join(self.images_dir, "*.jpg"))))
        if len(img_files) == 0:
            raise FileNotFoundError(f"No image files found in {self.images_dir}")

        target_files = [os.path.join(self.labels_dir, os.path.basename(img_file).replace(".jpg", ".txt")) for img_file in img_files]

        img_and_target_path_list = [(img_file, target_file) for img_file, target_file in zip(img_files, target_files) if os.path.exists(target_file)]
        if len(img_and_target_path_list) == 0:
            raise FileNotFoundError("No target files associated with the images were found")

        num_missing_files = len(img_files) - len(img_and_target_path_list)
        if num_missing_files > 0:
            logger.warning(f"{num_missing_files} label files were not loaded out of {len(img_files)} image files")

        self.img_and_target_path_list = img_and_target_path_list
        return len(self.img_and_target_path_list)

    def _load_annotation(self, sample_id: int) -> dict:
        """Load annotations for a given sample.

        :return: Annotation including:
                    - target in XYXY_LABEL format
                    - img_path
        """
        img_path, target_path = self.img_and_target_path_list[sample_id]
        with open(target_path, "r") as file:
            target = np.array([x.split() for x in file.read().splitlines()], dtype=np.float32)

        height, width = get_image_size_from_path(img_path)
        r = min(self.input_dim[1] / height, self.input_dim[0] / width)
        target[:, :4] *= r
        resized_img_shape = (int(height * r), int(width * r))

        return {"img_path": img_path, "target": target, "resized_img_shape": resized_img_shape}
