import os

import imagesize
import numpy as np
from typing import List, Optional

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataset
from super_gradients.training.exceptions.dataset_exceptions import DatasetValidationException

from super_gradients.training.datasets.data_formats import ConcatenatedTensorFormatConverter
from super_gradients.training.datasets.data_formats.default_formats import XYXY_LABEL, LABEL_NORMALIZED_CXCYWH

logger = get_logger(__name__)


class YoloV5FormattedDetectionDataset(DetectionDataset):
    """Base dataset to load ANY dataset that is with a similar structure to the YoloV5 dataset.

    **Note**: For compatibility reasons, the dataset returns labels in Coco format (XYXY_LABEL) and NOT in YoloV5 format (LABEL_CXCYWH).

    Output format: XYXY_LABEL (x, y, x, y, class_id)
    """

    def __init__(
        self,
        data_dir: str,
        images_dir: str,
        labels_dir: str,
        classes: List[str],
        class_ids_to_ignore: Optional[List[int]] = None,
        *args,
        **kwargs,
    ):
        """
        :param data_dir:                Where the data is stored.
        :param images_dir:              Name of the directory that includes all the images. Path relative to data_dir.
        :param labels_dir:              Name of the directory that includes all the labels. Path relative to data_dir.
        :param classes:                 List of class names.
        :param class_ids_to_ignore:     List of class ids to ignore in the dataset. By default, doesnt ignore any class.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_ids_to_ignore = class_ids_to_ignore or []
        self.classes = classes

        kwargs["target_fields"] = ["target"]
        kwargs["output_fields"] = ["image", "target"]
        kwargs["original_target_format"] = XYXY_LABEL  # We convert yolov5 format (LABEL_CXCYWH) to Coco format (XYXY_LABEL) when loading the annotation
        super().__init__(data_dir=data_dir, *args, **kwargs)

    @property
    def _all_classes(self) -> List[str]:
        return self.classes

    def _setup_data_source(self) -> int:
        """Initialize img_and_target_path_list and warn if label file is missing

        :return: number of images in the dataset
        """
        self.images_folder = os.path.join(self.data_dir, self.images_dir)
        self.labels_folder = os.path.join(self.data_dir, self.labels_dir)

        self.images_file_names = list(sorted(os.listdir(self.images_folder)))
        self.labels_file_names = list(sorted(os.listdir(self.labels_folder)))

        image_file_base_names = set(os.path.splitext(os.path.basename(image_file_name))[0] for image_file_name in self.images_file_names)
        label_file_base_names = set(os.path.splitext(os.path.basename(label_file_name))[0] for label_file_name in self.labels_file_names)
        if image_file_base_names != label_file_base_names:
            raise DatasetValidationException(f"image folder {self.images_folder} and label folder {self.labels_folder} include files that don't match")

        return len(self.images_file_names)

    def _load_annotation(self, sample_id: int) -> dict:
        """Load relevant information of a specific image.

        :param sample_id:   Sample_id in the dataset
        :return:            Dictionary with the following keys:
            - "target":             Target Bboxes (detection) in XYXY_LABEL format
            - "initial_img_shape":  Image (height, width)
            - "resized_img_shape":  Resides image (height, width)
            - "img_path":           Path to the associated image
        """
        image_path = os.path.join(self.images_folder, self.images_file_names[sample_id])
        label_path = os.path.join(self.labels_folder, self.labels_file_names[sample_id])

        image_width, image_height = imagesize.get(image_path)
        image_shape = (image_height, image_width)

        yolov5_format_target = parse_yolov5_label_file(label_path)

        converter = ConcatenatedTensorFormatConverter(input_format=LABEL_NORMALIZED_CXCYWH, output_format=XYXY_LABEL, image_shape=image_shape)
        target = converter(yolov5_format_target)

        # The base class includes a feature to resize the image, so we need to resize the target as well when self.input_dim is set.
        if self.input_dim is not None:
            r = min(self.input_dim[0] / image_height, self.input_dim[1] / image_width)
            target[:, :4] *= r
            resized_img_shape = (int(image_height * r), int(image_width * r))
        else:
            resized_img_shape = image_shape

        annotation = {
            "target": target,
            "initial_img_shape": image_shape,
            "resized_img_shape": resized_img_shape,
            "img_path": image_path,
            "id": np.array([sample_id]),
        }
        return annotation


def parse_yolov5_label_file(label_file_path: str) -> np.ndarray:
    """Parse a single label file in yolo v5 format.

    :return: np.ndarray of shape (n_labels, 5) in yolo v5 format (LABEL_NORMALIZED_CXCYWH)
    """
    with open(label_file_path, "r") as f:
        labels_txt = f.read()

    labels_yolo_v5_format = []
    for line in labels_txt.split("\n"):
        label_id, cx, cw, w, h = line.split(" ")
        labels_yolo_v5_format.append([int(label_id), float(cx), float(cw), float(w), float(h)])
    return np.array(labels_yolo_v5_format)
