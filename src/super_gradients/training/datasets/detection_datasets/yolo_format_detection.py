import os

import imagesize
import numpy as np
from typing import List, Optional, Tuple

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.media.image import is_image
from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataset
from super_gradients.training.datasets.data_formats import ConcatenatedTensorFormatConverter
from super_gradients.training.datasets.data_formats.default_formats import XYXY_LABEL, LABEL_NORMALIZED_CXCYWH

logger = get_logger(__name__)


class YoloDarknetFormatDetectionDataset(DetectionDataset):
    """Base dataset to load ANY dataset that is with a similar structure to the Yolo/Darknet dataset.

    **Note**: For compatibility reasons, the dataset returns labels in Coco format (XYXY_LABEL) and NOT in Yolo format (LABEL_CXCYWH).

    The dataset can have any structure, as long as `images_dir` and `labels_dir` inside `data_dir`.
    Each image is expected to have a file with the same name as the label.

    Example1:
        data_dir
        ├── images
        │      ├─ 0001.jpg
        │      ├─ 0002.jpg
        │      └─ ...
        └── labels
               ├─ 0001.txt
               ├─ 0002.txt
               └─ ...
        >> data_set = YoloDarknetFormatDetectionDataset(data_dir='<path-to>/data_dir', images_dir="images", labels_dir="labels", classes=[<to-fill>])

    Example2:
        data_dir
        ├── train
        │   ├── images
        │   │      ├─ 0001.jpg
        │   │      ├─ 0002.jpg
        │   │      └─ ...
        │   └── labels
        │          ├─ 0001.txt
        │          ├─ 0002.txt
        │          └─ ...
        └── val
            ├── images
            │      ├─ 434343.jpg
            │      ├─ 434344.jpg
            │      └─ ...
            └── labels
                   ├─ 434343.txt
                   ├─ 434344.txt
                   └─ ...

        >> train_set = YoloDarknetFormatDetectionDataset(
                data_dir='<path-to>/data_dir', images_dir="train/images", labels_dir="train/labels", classes=[<to-fill>]
            )
        >> val_set = YoloDarknetFormatDetectionDataset(
                data_dir='<path-to>/data_dir', images_dir="val/images", labels_dir="val/labels", classes=[<to-fill>]
            )

    Example3:
        data_dir
        ├── train
        │      ├─ 0001.jpg
        │      ├─ 0001.txt
        │      ├─ 0002.jpg
        │      ├─ 0002.txt
        │      └─ ...
        └── val
               ├─ 4343.jpg
               ├─ 4343.txt
               ├─ 4344.jpg
               ├─ 4344.txt
               └─ ...

        >> train_set = YoloDarknetFormatDetectionDataset(data_dir='<path-to>/data_dir', images_dir="train", labels_dir="train", classes=[<to-fill>])
        >> val_set = YoloDarknetFormatDetectionDataset(data_dir='<path-to>/data_dir', images_dir="val", labels_dir="val", classes=[<to-fill>])

    Each label file being in LABEL_NORMALIZED_CXCYWH format:
        0 0.33 0.33 0.50 0.44
        1 0.21 0.54 0.30 0.60
        ...


    Output format: XYXY_LABEL (x, y, x, y, class_id)
    """

    def __init__(
        self,
        data_dir: str,
        images_dir: str,
        labels_dir: str,
        classes: List[str],
        class_ids_to_ignore: Optional[List[int]] = None,
        ignore_invalid_labels: bool = True,
        show_all_warnings: bool = False,
        *args,
        **kwargs,
    ):
        """
        :param data_dir:                Where the data is stored.
        :param images_dir:              Local path to directory that includes all the images. Path relative to `data_dir`. Can be the same as `labels_dir`.
        :param labels_dir:              Local path to directory that includes all the labels. Path relative to `data_dir`. Can be the same as `images_dir`.
        :param classes:                 List of class names.
        :param class_ids_to_ignore:     List of class ids to ignore in the dataset. By default, doesnt ignore any class.
        :param ignore_invalid_labels:   Whether to ignore labels that fail to be parsed. If True ignores and logs a warning, otherwise raise an error.
        :param show_all_warnings:       Whether to show every yolo format parser warnings or not.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_ids_to_ignore = class_ids_to_ignore or []
        self.classes = classes
        self.ignore_invalid_labels = ignore_invalid_labels
        self.show_all_warnings = show_all_warnings

        kwargs["target_fields"] = ["target"]
        kwargs["output_fields"] = ["image", "target"]
        kwargs["original_target_format"] = XYXY_LABEL  # We convert yolo format (LABEL_CXCYWH) to Coco format (XYXY_LABEL) when loading the annotation
        super().__init__(data_dir=data_dir, show_all_warnings=show_all_warnings, *args, **kwargs)

    @property
    def _all_classes(self) -> List[str]:
        return self.classes

    def _setup_data_source(self) -> int:
        """Initialize img_and_target_path_list and warn if label file is missing

        :return: number of images in the dataset
        """
        self.images_folder = os.path.join(self.data_dir, self.images_dir)
        self.labels_folder = os.path.join(self.data_dir, self.labels_dir)

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
        valid_base_names = unique_image_file_base_names & unique_label_file_base_names
        if len(valid_base_names) != len(all_images_file_names):
            logger.warning(
                f"As a consequence, "
                f"{len(valid_base_names)}/{len(all_images_file_names)} images and "
                f"{len(valid_base_names)}/{len(all_labels_file_names)} label files will be used."
            )

        self.images_file_names = []
        self.labels_file_names = []
        for image_full_name in all_images_file_names:
            base_name = remove_file_extension(image_full_name)
            if base_name in valid_base_names:
                self.images_file_names.append(image_full_name)
                self.labels_file_names.append(base_name + ".txt")
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

        yolo_format_target, invalid_labels = self._parse_yolo_label_file(
            label_file_path=label_path,
            ignore_invalid_labels=self.ignore_invalid_labels,
            show_warnings=self.show_all_warnings,
        )

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
            "n_invalid_labels": len(invalid_labels),
        }
        return annotation

    @staticmethod
    def _parse_yolo_label_file(label_file_path: str, ignore_invalid_labels: bool = True, show_warnings: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Parse a single label file in yolo format.

        #TODO: Add support for additional fields (with ConcatenatedTensorFormat)
        :param label_file_path:         Path to the label file in yolo format.
        :param ignore_invalid_labels:   Whether to ignore labels that fail to be parsed. If True ignores and logs a warning, otherwise raise an error.
        :param show_warnings:           Whether to show the warnings or not.

        :return:
            - labels:           np.ndarray of shape (n_labels, 5) in yolo format (LABEL_NORMALIZED_CXCYWH)
            - invalid_labels:   List of lines that failed to be parsed
        """
        with open(label_file_path, "r") as f:
            lines = f.readlines()

        labels_yolo_format, invalid_labels = [], []
        for line in filter(lambda x: x != "\n", lines):
            try:
                label_id, cx, cw, w, h = line.split()
                labels_yolo_format.append([int(label_id), float(cx), float(cw), float(w), float(h)])
            except Exception as e:
                if ignore_invalid_labels:
                    invalid_labels.append(line)
                    if show_warnings:
                        logger.warning(f"Line `{line}` of file {label_file_path} will be ignored because not in LABEL_NORMALIZED_CXCYWH format: {e}")
                else:
                    raise e
        return np.array(labels_yolo_format) if labels_yolo_format else np.zeros((0, 5)), invalid_labels
