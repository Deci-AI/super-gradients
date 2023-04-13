import os

import imagesize
import numpy as np
from typing import List, Optional

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataset

from super_gradients.training.datasets.data_formats import ConcatenatedTensorFormatConverter
from super_gradients.training.datasets.data_formats.default_formats import XYXY_LABEL, LABEL_NORMALIZED_CXCYWH

logger = get_logger(__name__)


class YoloDarknetFormatDetectionDataset(DetectionDataset):
    """Base dataset to load ANY dataset that is with a similar structure to the Yolo/Darknet dataset.

    **Note**: For compatibility reasons, the dataset returns labels in Coco format (XYXY_LABEL) and NOT in Yolo format (LABEL_CXCYWH).

    The dataset can have any structure, as long as `images_dir_name` are `labels_dir_name` inside `data_dir`.
    Each image is expected to have a file with the same name as the label.

    Example1:
        data_dir
        ├── images_dir_name
        │      ├─ 0001.jpg
        │      ├─ 0002.jpg
        │      └─ ...
        └── labels_dir_name
               ├─ 0001.txt
               ├─ 0002.txt
               └─ ...

    Example2:
        data_dir
        ├── train
        │   ├── images_dir_name
        │   │      ├─ 0001.jpg
        │   │      ├─ 0002.jpg
        │   │      └─ ...
        │   └── labels_dir_name
        │          ├─ 0001.txt
        │          ├─ 0002.txt
        │          └─ ...
        └── val
            ├── images_dir_name
            │      ├─ 434343.jpg
            │      ├─ 434344.jpg
            │      └─ ...
            └── labels_dir_name
                   ├─ 434343.txt
                   ├─ 434344.txt
                   └─ ...


    Each label file being in LABEL_NORMALIZED_CXCYWH format:
        0 0.33 0.33 0.50 0.44
        1 0.21 0.54 0.30 0.60
        ...


    Output format: XYXY_LABEL (x, y, x, y, class_id)
    """

    def __init__(
        self,
        data_dir: str,
        images_dir_name: str,
        labels_dir_name: str,
        classes: List[str],
        class_ids_to_ignore: Optional[List[int]] = None,
        *args,
        **kwargs,
    ):
        """
        :param data_dir:                Where the data is stored.
        :param images_dir_name:         Name of the directory that includes all the images. Path relative to `data_dir`. Can be the same as `labels_dir_name`.
        :param labels_dir_name:         Name of the directory that includes all the labels. Path relative to `data_dir`. Can be the same as `images_dir_name`.
        :param classes:                 List of class names.
        :param class_ids_to_ignore:     List of class ids to ignore in the dataset. By default, doesnt ignore any class.
        """
        self.images_dir_name = images_dir_name
        self.labels_dir_name = labels_dir_name
        self.class_ids_to_ignore = class_ids_to_ignore or []
        self.classes = classes

        kwargs["target_fields"] = ["target"]
        kwargs["output_fields"] = ["image", "target"]
        kwargs["original_target_format"] = XYXY_LABEL  # We convert yolo format (LABEL_CXCYWH) to Coco format (XYXY_LABEL) when loading the annotation
        super().__init__(data_dir=data_dir, *args, **kwargs)

    @property
    def _all_classes(self) -> List[str]:
        return self.classes

    def _setup_data_source(self) -> int:
        """Initialize img_and_target_path_list and warn if label file is missing

        :return: number of images in the dataset
        """
        self.images_folder = os.path.join(self.data_dir, self.images_dir_name)
        self.labels_folder = os.path.join(self.data_dir, self.labels_dir_name)

        all_images_file_names = list(image_name for image_name in os.listdir(self.images_folder) if is_image(image_name))
        all_labels_file_names = list(label_name for label_name in os.listdir(self.labels_folder) if label_name.endswith(".txt"))

        remove_file_extension = lambda file_name: os.path.splitext(os.path.basename(file_name))[0]
        unique_image_file_base_names = set(remove_file_extension(image_file_name) for image_file_name in all_images_file_names)
        unique_label_file_base_names = set(remove_file_extension(label_file_name) for label_file_name in all_labels_file_names)

        images_not_in_labels = unique_image_file_base_names - unique_label_file_base_names
        if images_not_in_labels:
            logger.warning(f"{len(images_not_in_labels)} images are note associated to any label file")

        labels_not_in_images = unique_label_file_base_names - unique_image_file_base_names
        if labels_not_in_images:
            logger.warning(f"{len(labels_not_in_images)} label files are not associated to any image.")

        # Only keep names that are in both the images and the labels
        valid_base_names = list(unique_image_file_base_names & unique_label_file_base_names)
        if len(valid_base_names) != len(all_images_file_names):
            logger.warning(
                f"As a consequence, "
                f"{len(valid_base_names)}/{len(all_images_file_names)} images and "
                f"{len(valid_base_names)}/{len(all_labels_file_names)} label files will be used."
            )

        self.images_file_names = list(
            sorted(image_full_name for image_full_name in all_images_file_names if remove_file_extension(image_full_name) in valid_base_names)
        )
        self.labels_file_names = list(
            sorted(label_full_name for label_full_name in all_labels_file_names if remove_file_extension(label_full_name) in valid_base_names)
        )
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

        yolo_format_target = parse_yolo_label_file(label_path)

        converter = ConcatenatedTensorFormatConverter(input_format=LABEL_NORMALIZED_CXCYWH, output_format=XYXY_LABEL, image_shape=image_shape)
        target = converter(yolo_format_target)

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


def parse_yolo_label_file(label_file_path: str) -> np.ndarray:
    """Parse a single label file in yolo format.

    #TODO: Add support for additional fields (with ConcatenatedTensorFormat)

    :return: np.ndarray of shape (n_labels, 5) in yolo format (LABEL_NORMALIZED_CXCYWH)
    """
    with open(label_file_path, "r") as f:
        labels_txt = f.read()

    labels_yolo_format = []
    for line in labels_txt.split("\n"):
        label_id, cx, cw, w, h = line.split(" ")
        labels_yolo_format.append([int(label_id), float(cx), float(cw), float(w), float(h)])
    return np.array(labels_yolo_format)


def is_image(filename: str) -> bool:
    IMG_EXTENSIONS = ("bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm")
    return filename.split(".")[-1].lower() in IMG_EXTENSIONS
