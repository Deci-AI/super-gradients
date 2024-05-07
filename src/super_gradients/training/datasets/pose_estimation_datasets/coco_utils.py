import dataclasses
import json
import numbers
import os
import pprint
from enum import Enum
from typing import Tuple, List, Dict

import cv2
import numpy as np

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy_inplace

logger = get_logger(__name__)

__all__ = ["CrowdAnnotationActionEnum", "KeypointsAnnotation", "parse_coco_into_keypoints_annotations"]


class CrowdAnnotationActionEnum(str, Enum):
    """
    Enum that contains possible actions to take for crowd annotations.
    """

    DROP_SAMPLE = "drop_sample"
    DROP_ANNOTATION = "drop_annotation"
    MASK_AS_NORMAL = "mask_as_normal"
    NO_ACTION = "no_action"


@dataclasses.dataclass
class KeypointsAnnotation:
    image_id: int
    image_path: str
    image_width: int
    image_height: int

    ann_boxes_xyxy: np.ndarray
    ann_is_crowd: np.ndarray
    ann_areas: np.ndarray
    ann_keypoints: np.ndarray
    ann_segmentations: np.ndarray


def parse_coco_into_keypoints_annotations(
    ann: str,
    image_path_prefix=None,
    crowd_annotations_action=CrowdAnnotationActionEnum.NO_ACTION,
    remove_duplicate_annotations: bool = False,
) -> Tuple[str, Dict, List[KeypointsAnnotation]]:
    """
    Load COCO keypoints dataset from annotation file.
    :param ann: A path to the JSON annotation file in COCO format.
    :param image_path_prefix:   A prefix to add to the image paths in the annotation file.
    :return:                    Tuple (class_names, annotations) where class_names is a list of class names
                                (respecting include_classes/exclude_classes/class_ids_to_ignore) and
                                annotations is a list of DetectionAnnotation objects.
    """
    with open(ann, "r") as f:
        coco = json.load(f)

    if len(coco["categories"]) != 1:
        raise ValueError("Dataset must contain exactly one category")

    # Extract class names and class ids
    category_name = coco["categories"][0]["name"]
    keypoints = coco["categories"][0]["keypoints"]
    num_keypoints = len(keypoints)

    # Extract box annotations
    ann_box_xyxy = xywh_to_xyxy_inplace(np.array([annotation["bbox"] for annotation in coco["annotations"]], dtype=np.float32), image_shape=None)
    ann_keypoints = np.stack([np.array(annotation["keypoints"], dtype=np.float32).reshape(num_keypoints, 3) for annotation in coco["annotations"]])
    ann_iscrowd = np.array([annotation["iscrowd"] for annotation in coco["annotations"]], dtype=bool)
    ann_image_ids = np.array([annotation["image_id"] for annotation in coco["annotations"]], dtype=int)
    ann_segmentations = np.array([annotation["segmentation"] for annotation in coco["annotations"]], dtype=np.object_)

    # We check whether the area is present in the annotations. If it does we use it, otherwise we compute it from the bbox.
    if "area" in coco["annotations"][0]:
        ann_areas = np.array([annotation["area"] for annotation in coco["annotations"]], dtype=np.float32)
    else:
        # Compute area from box
        # A multiplier of 0.53 is a heuristic from pycocotools to approximate the area of the pose instance
        # from the area of the bounding box.
        ann_areas = np.prod(ann_box_xyxy[:, 2:] - ann_box_xyxy[:, :2], axis=-1) * 0.53

    # Extract image stuff
    img_ids = np.array([img["id"] for img in coco["images"]], dtype=int)
    img_paths = np.array([img["file_name"] if "file_name" in img else "{:012}".format(img["id"]) + ".jpg" for img in coco["images"]], dtype=str)
    img_width_height = np.array([(img["width"], img["height"]) for img in coco["images"]], dtype=int)

    annotations = []

    if crowd_annotations_action == CrowdAnnotationActionEnum.MASK_AS_NORMAL:
        ann_iscrowd = np.zeros_like(ann_iscrowd, dtype=bool)
    elif crowd_annotations_action == CrowdAnnotationActionEnum.DROP_ANNOTATION:
        ann_box_xyxy = ann_box_xyxy[~ann_iscrowd]
        ann_keypoints = ann_keypoints[~ann_iscrowd]
        ann_areas = ann_areas[~ann_iscrowd]
        ann_segmentations = ann_segmentations[~ann_iscrowd]
        ann_image_ids = ann_image_ids[~ann_iscrowd]
        ann_iscrowd = ann_iscrowd[~ann_iscrowd]

    for img_id, image_path, (image_width, image_height) in zip(img_ids, img_paths, img_width_height):
        mask = ann_image_ids == img_id

        if image_path_prefix is not None:
            image_path = os.path.join(image_path_prefix, image_path)

        ann = KeypointsAnnotation(
            image_id=img_id,
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            ann_boxes_xyxy=ann_box_xyxy[mask],
            ann_is_crowd=ann_iscrowd[mask],
            ann_areas=ann_areas[mask],
            ann_keypoints=ann_keypoints[mask],
            ann_segmentations=ann_segmentations[mask],
        )

        if remove_duplicate_annotations:
            joints = ann.ann_keypoints[:, :, :2]
            gt_joints1 = np.expand_dims(joints, axis=0)  # [1, Num_people, Num_joints, 2]
            gt_joints2 = np.expand_dims(joints, axis=1)  # [Num_people, 1, Num_joints, 2]
            diff = np.sqrt(np.sum((gt_joints1 - gt_joints2) ** 2, axis=-1))  # [Num_people, Num_people, Num_joints]
            diffmean = np.mean(diff, axis=-1)

            duplicate_mask = np.triu(diffmean < 2, k=1)
            duplicate_indexes_i, duplicate_indexes_j = np.nonzero(duplicate_mask)
            keep_mask = np.ones(len(ann.ann_boxes_xyxy), dtype=bool)
            for i, j in zip(duplicate_indexes_i, duplicate_indexes_j):
                keep_mask[j] = False

            ann.ann_boxes_xyxy = ann.ann_boxes_xyxy[keep_mask]
            ann.ann_keypoints = ann.ann_keypoints[keep_mask]
            ann.ann_areas = ann.ann_areas[keep_mask]
            ann.ann_segmentations = ann.ann_segmentations[keep_mask]
            ann.ann_is_crowd = ann.ann_is_crowd[keep_mask]

        if crowd_annotations_action == CrowdAnnotationActionEnum.DROP_SAMPLE:
            if ann.ann_is_crowd.any():
                continue

        annotations.append(ann)

    return category_name, keypoints, annotations


def poly2mask(points: List, image: np.ndarray):
    points = (np.array(points)).reshape(-1, 2).astype(int)
    cv2.fillPoly(image, [points], 1)
    return image


def segmentation2mask(segmentation, image_shape: Tuple[int, int]):
    """
    Decode segmentation annotation into binary mask
    :param segmentation: Input segmentation annotation. Can come in many forms:
                         -
    :param image_shape:
    :return:
    """
    m = np.zeros(image_shape, dtype=np.uint8)

    if isinstance(segmentation, list) and len(segmentation):
        if isinstance(segmentation[0], numbers.Number):
            if len(segmentation) == 4:
                # box?
                unsupported_input_repr = pprint.pformat(segmentation)
                raise ValueError(
                    "Box encoding is not supported yet.\n"
                    "Please open an issue on GitHub (https://github.com/Deci-AI/super-gradients/issues) and attach the following information:\n"
                    "```python\n"
                    f"image_shape = {image_shape}\n"
                    f"segmentation = {unsupported_input_repr}\n"
                    "```python\n"
                )
            else:
                poly2mask(segmentation, m)
        else:
            for seg_i in segmentation:
                poly2mask(seg_i, m)
    elif isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
        rle = segmentation["counts"]
        m = rle2mask(rle, image_shape)
    else:
        unsupported_input_repr = pprint.pformat(segmentation)
        raise ValueError(
            "Unknown segmentation format\n"
            "Please open an issue on GitHub (https://github.com/Deci-AI/super-gradients/issues) and attach the following information:\n"
            "```python\n"
            f"image_shape = {image_shape}\n"
            f"segmentation = {unsupported_input_repr}\n"
            "```python\n"
        )
    return m


def rle2mask(rle: np.ndarray, image_shape: Tuple[int, int]):
    """
    Convert RLE to binary mask
    :param rle: A string containing RLE-encoded mask
    :param image_shape: Output image shape (rows, cols)
    :return: A decoded binary mask
    """
    rle = np.array(rle, dtype=int)

    value = 0
    start = 0
    img = np.zeros(image_shape[0] * image_shape[1], dtype=np.uint8)
    for offset in rle:
        img[start : start + offset] = value
        start += offset
        value = 1 - value

    return img.reshape(*reversed(image_shape)).T
