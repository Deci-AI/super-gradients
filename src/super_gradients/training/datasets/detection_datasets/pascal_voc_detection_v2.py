import os
import glob
from typing import Dict

import numpy as np

from super_gradients.training.datasets.detection_datasets.detection_dataset_v2 import DetectionDataSetV2
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat
from super_gradients.common.abstractions.abstract_logger import get_logger


logger = get_logger(__name__)

PASCAL_VOC_2012_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    'tvmonitor'
]


class PascalVOCDetectionDataSetV2(DetectionDataSetV2):
    """
    PascalVOCDetectionDataSetV2 - Detection Data Set Class pascal_voc Data Set

    You can check the implementation of DetectionDataSetV2 to better understand the flow
    """

    def __init__(self, data_dir: str, images_sub_directory: str, *args, **kwargs):
        self.data_dir = data_dir
        self.images_sub_directory = images_sub_directory
        self.img_and_annotation_tuples_list = self._get_img_and_annotation_tuples_list()

        kwargs['n_available_samples'] = len(self.img_and_annotation_tuples_list)
        kwargs['all_classes_list'] = PASCAL_VOC_2012_CLASSES
        kwargs['target_format'] = DetectionTargetsFormat.XYXY_LABEL
        super().__init__(*args, **kwargs)

    def _get_img_and_annotation_tuples_list(self):
        """Initialize img_and_annotation_tuples_list and warn if label file is missing"""
        img_files_folder = self.data_dir + self.images_sub_directory
        img_files = glob.glob(img_files_folder + "*.jpg")
        assert len(img_files) > 0, f"No image file found at {img_files_folder}"

        annotation_files = [img_file.replace("images", "labels").replace(".jpg", ".txt") for img_file in img_files]

        img_and_annotation_tuples_list = [
            (img_file, annotation_file)
            for img_file, annotation_file in zip(img_files, annotation_files)
            if os.path.exists(annotation_file)
        ]

        num_missing_files = len(img_files) - len(img_and_annotation_tuples_list)
        if num_missing_files > 0:
            logger.warning(f'{num_missing_files} labels files were not loaded our of {len(img_files)} image files')
        return img_and_annotation_tuples_list

    def _load_annotation(self, sample_id: int) -> Dict[str, np.array]:
        """Load from disk annotation, which is only made of the image target.

        :return target: target in XYXY_LABEL format
        """
        img_path, annotation_path = self.img_and_annotation_tuples_list[sample_id]
        with open(annotation_path, 'r') as targets_file:
            target = np.array([x.split() for x in targets_file.read().splitlines()], dtype=np.float32)
        return {"target": target, "img_path": img_path}
