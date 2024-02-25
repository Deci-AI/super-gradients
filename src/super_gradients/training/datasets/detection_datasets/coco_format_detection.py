import copy
import dataclasses
import json
import os

import numpy as np
from typing import List, Optional, Tuple

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.exceptions.dataset_exceptions import DatasetValidationException, ParameterMismatchException
from super_gradients.common.deprecate import deprecated_parameter
from super_gradients.common.registry import register_dataset
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy_inplace
from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataset
from super_gradients.training.datasets.data_formats.default_formats import XYXY_LABEL
from super_gradients.training.utils.detection_utils import change_bbox_bounds_for_image_size

logger = get_logger(__name__)


@register_dataset("COCOFormatDetectionDataset")
class COCOFormatDetectionDataset(DetectionDataset):
    """Base dataset to load ANY dataset that is with a similar structure to the COCO dataset.
    - Annotation file (.json). It has to respect the exact same format as COCO, for both the json schema and the bbox format (xywh).
    - One folder with all the images.

    Output format: (x, y, x, y, class_id)
    """

    @deprecated_parameter(
        "tight_box_rotation",
        deprecated_since="3.7.0",
        removed_from="3.8.0",
        reason="Support of `tight_box_rotation` has been removed. This parameter has no effect anymore.",
    )
    def __init__(
        self,
        data_dir: str,
        json_annotation_file: str,
        images_dir: str,
        with_crowd: bool = True,
        class_ids_to_ignore: Optional[List[int]] = None,
        tight_box_rotation=None,
        *args,
        **kwargs,
    ):
        """
        :param data_dir:                Where the data is stored.
        :param json_annotation_file:    Name of the coco json file. Path can be either absolute, or relative to data_dir.
        :param images_dir:              Name of the directory that includes all the images. Path relative to data_dir.
        :param with_crowd:              Add the crowd groundtruths to __getitem__
        :param class_ids_to_ignore:     List of class ids to ignore in the dataset. By default, doesnt ignore any class.
        :param tight_box_rotation:      This parameter is deprecated and will be removed in a SuperGradients 3.8.
        """
        if tight_box_rotation is not None:
            logger.warning(
                "Parameter `tight_box_rotation` is deprecated and will be removed in a SuperGradients 3.8." "Please remove this parameter from your code."
            )
        self.images_dir = images_dir
        self.json_annotation_file = json_annotation_file
        self.with_crowd = with_crowd
        self.class_ids_to_ignore = class_ids_to_ignore or []

        target_fields = ["target", "crowd_target"] if self.with_crowd else ["target"]
        kwargs["target_fields"] = target_fields
        kwargs["output_fields"] = ["image", *target_fields]
        kwargs["original_target_format"] = XYXY_LABEL
        super().__init__(data_dir=data_dir, *args, **kwargs)

        if len(self.original_classes) != len(self.all_classes_list):
            if set(self.all_classes_list).issubset(set(self.original_classes)):
                raise ParameterMismatchException(
                    "Parameter `all_classes_list` contains a subset of classes from dataset JSON. "
                    "Please use `class_inclusion_list` to train with reduced number of classes",
                )
            else:
                raise DatasetValidationException(
                    "Number of classes in dataset JSON do not match with number of classes in all_classes_list parameter. "
                    "Most likely this indicates an error in your all_classes_list parameter"
                )

    def _setup_data_source(self) -> int:
        """
        Parse COCO annotation file
        :return: Number of images in annotation JSON
        """
        if os.path.isabs(self.json_annotation_file):
            annotation_file_path = self.json_annotation_file
        else:
            annotation_file_path = os.path.join(self.data_dir, self.json_annotation_file)
        if not os.path.exists(annotation_file_path):
            raise ValueError("Could not find annotation file under " + str(annotation_file_path))

        all_class_names, annotations = parse_coco_into_detection_annotations(
            annotation_file_path,
            exclude_classes=None,
            include_classes=None,
            # This parameter exists solely for the purpose of keeping the backward compatibility with the old code.
            # Once we refactor base dataset, we can remove this parameter and use only exclude_classes/include_classes
            # at parsing time instead.
            class_ids_to_ignore=self.class_ids_to_ignore,
            image_path_prefix=os.path.join(self.data_dir, self.images_dir),
        )

        self.original_classes = list(all_class_names)
        self.classes = copy.deepcopy(self.original_classes)
        self._annotations = annotations
        return len(annotations)

    @property
    def _all_classes(self) -> List[str]:
        return self.original_classes

    def _load_annotation(self, sample_id: int) -> dict:
        """
        Load relevant information of a specific image.

        :param sample_id:               Sample_id in the dataset
        :return target:                 Target Bboxes (detection) in XYXY_LABEL format
        :return crowd_target:           Crowd target Bboxes (detection) in XYXY_LABEL format
        :return target_segmentation:    Segmentation
        :return initial_img_shape:      Image (height, width)
        :return resized_img_shape:      Resides image (height, width)
        :return img_path:               Path to the associated image
        """

        annotation = self._annotations[sample_id]

        width = annotation.image_width
        height = annotation.image_height

        # Make a copy of the annotations, so that we can modify them
        boxes_xyxy = change_bbox_bounds_for_image_size(annotation.ann_boxes_xyxy, img_shape=(height, width), inplace=False)
        iscrowd = annotation.ann_is_crowd.copy()
        labels = annotation.ann_labels.copy()

        # Exclude boxes with invalid dimensions (x1 > x2 or y1 > y2)
        mask = np.logical_and(boxes_xyxy[:, 2] >= boxes_xyxy[:, 0], boxes_xyxy[:, 3] >= boxes_xyxy[:, 1])
        boxes_xyxy = boxes_xyxy[mask]
        iscrowd = iscrowd[mask]
        labels = labels[mask]

        # Currently, the base class includes a feature to resize the image, so we need to resize the target as well when self.input_dim is set.
        initial_img_shape = (height, width)
        if self.input_dim is not None:
            scale_factor = min(self.input_dim[0] / height, self.input_dim[1] / width)
            resized_img_shape = (int(height * scale_factor), int(width * scale_factor))
        else:
            resized_img_shape = initial_img_shape
            scale_factor = 1

        targets = np.concatenate([boxes_xyxy[~iscrowd] * scale_factor, labels[~iscrowd, None]], axis=1).astype(np.float32)
        crowd_targets = np.concatenate([boxes_xyxy[iscrowd] * scale_factor, labels[iscrowd, None]], axis=1).astype(np.float32)

        annotation = {
            "target": targets,
            "crowd_target": crowd_targets,
            "initial_img_shape": initial_img_shape,
            "resized_img_shape": resized_img_shape,
            "img_path": annotation.image_path,
        }
        return annotation


@dataclasses.dataclass
class DetectionAnnotation:
    image_id: int
    image_path: str
    image_width: int
    image_height: int

    # Bounding boxes in XYXY format
    ann_boxes_xyxy: np.ndarray
    ann_is_crowd: np.ndarray
    ann_labels: np.ndarray


def parse_coco_into_detection_annotations(
    ann: str,
    exclude_classes: Optional[List[str]] = None,
    include_classes: Optional[List[str]] = None,
    class_ids_to_ignore: Optional[List[int]] = None,
    image_path_prefix=None,
) -> Tuple[List[str], List[DetectionAnnotation]]:
    """
    Load COCO detection dataset from annotation file.
    :param ann: A path to the JSON annotation file in COCO format.
    :param exclude_classes: List of classes to exclude from the dataset. All other classes will be included.
                                This parameter is mutually exclusive with include_classes and class_ids_to_ignore.

    :param include_classes:     List of classes to include in the dataset. All other classes will be excluded.
                                This parameter is mutually exclusive with exclude_classes and class_ids_to_ignore.
    :param class_ids_to_ignore: List of category ids to ignore in the dataset. All other classes will be included.
                                This parameter added for the purpose of backward compatibility with the class_ids_to_ignore
                                argument of COCOFormatDetectionDataset but will be
                                removed in future in favor of include_classes/exclude_classes.
                                This parameter is mutually exclusive with exclude_classes and include_classes.
    :param image_path_prefix:   A prefix to add to the image paths in the annotation file.
    :return:                    Tuple (class_names, annotations) where class_names is a list of class names
                                (respecting include_classes/exclude_classes/class_ids_to_ignore) and
                                annotations is a list of DetectionAnnotation objects.
    """
    with open(ann, "r") as f:
        coco = json.load(f)

    # Extract class names and class ids
    category_ids = np.array([category["id"] for category in coco["categories"]], dtype=int)
    category_names = np.array([category["name"] for category in coco["categories"]], dtype=str)

    # Extract box annotations
    ann_box_xyxy = xywh_to_xyxy_inplace(np.array([annotation["bbox"] for annotation in coco["annotations"]], dtype=np.float32), image_shape=None)

    ann_category_id = np.array([annotation["category_id"] for annotation in coco["annotations"]], dtype=int)
    ann_iscrowd = np.array([annotation["iscrowd"] for annotation in coco["annotations"]], dtype=bool)
    ann_image_ids = np.array([annotation["image_id"] for annotation in coco["annotations"]], dtype=int)

    # Extract image stuff
    img_ids = np.array([img["id"] for img in coco["images"]], dtype=int)
    img_paths = np.array([img["file_name"] if "file_name" in img else "{:012}".format(img["id"]) + ".jpg" for img in coco["images"]], dtype=str)
    img_width_height = np.array([(img["width"], img["height"]) for img in coco["images"]], dtype=int)

    # Now, we can drop the annotations that belongs to the excluded classes
    if int(class_ids_to_ignore is not None) + int(exclude_classes is not None) + int(include_classes is not None) > 1:
        raise ValueError("Only one of exclude_classes, class_ids_to_ignore or include_classes can be specified")
    elif exclude_classes is not None:
        if len(exclude_classes) != len(set(exclude_classes)):
            raise ValueError("The excluded classes must be unique")
        classes_not_in_dataset = set(exclude_classes).difference(set(category_names))
        if len(classes_not_in_dataset) > 0:
            raise ValueError(f"One or more of the excluded classes does not exist in the dataset: {classes_not_in_dataset}")
        keep_classes_mask = np.isin(category_names, exclude_classes, invert=True)
    elif class_ids_to_ignore is not None:
        if len(class_ids_to_ignore) != len(set(class_ids_to_ignore)):
            raise ValueError("The ignored classes must be unique")
        classes_not_in_dataset = set(class_ids_to_ignore).difference(set(category_ids))
        if len(classes_not_in_dataset) > 0:
            raise ValueError(f"One or more of the ignored classes does not exist in the dataset: {classes_not_in_dataset}")
        keep_classes_mask = np.isin(category_ids, class_ids_to_ignore, invert=True)
    elif include_classes is not None:
        if len(include_classes) != len(set(include_classes)):
            raise ValueError("The included classes must be unique")
        classes_not_in_dataset = set(include_classes).difference(set(category_names))
        if len(classes_not_in_dataset) > 0:
            raise ValueError(f"One or more of the included classes does not exist in the dataset: {classes_not_in_dataset}")
        keep_classes_mask = np.isin(category_names, include_classes)
    else:
        keep_classes_mask = None

    if keep_classes_mask is not None:
        category_ids = category_ids[keep_classes_mask]
        category_names = category_names[keep_classes_mask]

        keep_anns_mask = np.isin(ann_category_id, category_ids)
        ann_category_id = ann_category_id[keep_anns_mask]

    # category_ids can be non-sequential and not ordered
    num_categories = len(category_ids)

    # Make sequential
    order = np.argsort(category_ids, kind="stable")
    category_ids = category_ids[order]  #
    category_names = category_names[order]

    # Remap category ids to be in range [0, num_categories)
    class_label_table = np.zeros(np.max(category_ids) + 1, dtype=int) - 1
    new_class_ids = np.arange(num_categories, dtype=int)
    class_label_table[category_ids] = new_class_ids

    # Remap category ids in annotations
    ann_category_id = class_label_table[ann_category_id]
    if (ann_category_id < 0).any():
        raise ValueError("Some annotations have class ids that are not in the list of classes. This probably indicates a bug in the annotation file")

    annotations = []

    for img_id, image_path, (image_width, image_height) in zip(img_ids, img_paths, img_width_height):
        mask = ann_image_ids == img_id

        if image_path_prefix is not None:
            image_path = os.path.join(image_path_prefix, image_path)

        ann = DetectionAnnotation(
            image_id=img_id,
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            ann_boxes_xyxy=ann_box_xyxy[mask],
            ann_is_crowd=ann_iscrowd[mask],
            ann_labels=ann_category_id[mask],
        )
        annotations.append(ann)

    return category_names, annotations
