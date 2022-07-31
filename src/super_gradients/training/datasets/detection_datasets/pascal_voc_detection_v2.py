import os
import glob
from typing import List, Tuple

import numpy as np

from super_gradients.training.datasets.detection_datasets.detection_dataset_v2 import DetectionDataSetV2
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.datasets_conf import PASCAL_VOC_2012_CLASSES_LIST

logger = get_logger(__name__)


class PascalVOCDetectionDataSetV2(DetectionDataSetV2):
    """Dataset for Pascal VOC object detection"""

    def __init__(self, data_dir: str, images_sub_directory: str, *args, **kwargs):
        """Dataset for Pascal VOC object detection

        :param data_dir:                Where the data is stored
        :param images_sub_directory:    Sub directory of data_dir that includes images.
        """
        self.data_dir = data_dir
        self.images_sub_directory = images_sub_directory
        self.img_and_target_path_list = self._get_img_and_target_path_list()

        kwargs['n_available_samples'] = len(self.img_and_target_path_list)
        kwargs['all_classes_list'] = PASCAL_VOC_2012_CLASSES_LIST
        kwargs['target_format'] = DetectionTargetsFormat.XYXY_LABEL
        super().__init__(*args, **kwargs)

    def _get_img_and_target_path_list(self) -> List[Tuple[str, str]]:
        """Initialize img_and_target_path_list and warn if label file is missing

        :return: List of tuples made of (img_path,target_path)
        """
        img_files_folder = self.data_dir + self.images_sub_directory
        img_files = glob.glob(img_files_folder + "*.jpg")
        if len(img_files) == 0:
            raise FileNotFoundError(f"No image file found at {img_files_folder}")

        target_files = [img_file.replace("images", "labels").replace(".jpg", ".txt") for img_file in img_files]

        img_and_target_path_list = [(img_file, target_file)
                                    for img_file, target_file in zip(img_files, target_files)
                                    if os.path.exists(target_file)]
        if len(img_and_target_path_list) == 0:
            raise FileNotFoundError("No target file associated to the images was found")

        num_missing_files = len(img_files) - len(img_and_target_path_list)
        if num_missing_files > 0:
            logger.warning(f'{num_missing_files} labels files were not loaded our of {len(img_files)} image files')
        return img_and_target_path_list

    def _load_annotation(self, sample_id: int) -> dict:
        """Load annotations associated to a specific sample.

        :return: Annotation including:
                    - target in XYXY_LABEL format
                    - img_path
        """
        img_path, target_path = self.img_and_target_path_list[sample_id]
        with open(target_path, 'r') as targets_file:
            target = np.array([x.split() for x in targets_file.read().splitlines()], dtype=np.float32)
        return {"target": target, "img_path": img_path}
