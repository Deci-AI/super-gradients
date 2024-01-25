import dataclasses
import json
import os
from enum import Enum
from typing import Tuple, List, Dict

import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy_inplace

logger = get_logger(__name__)

__all__ = [
    "CrowdAnnotationActionEnum",
    "check_keypoints_outside_image",
    "check_for_duplicate_annotations",
    "make_keypoints_outside_image_invisible",
    "remove_duplicate_annotations",
    "remove_crowd_annotations",
    "remove_samples_with_crowd_annotations",
]


class CrowdAnnotationActionEnum(str, Enum):
    """
    Enum that contains possible actions to take for crowd annotations.
    """

    DROP_SAMPLE = "drop_sample"
    DROP_ANNOTATION = "drop_annotation"
    MASK_AS_NORMAL = "mask_as_normal"
    NO_ACTION = "no_action"


def check_keypoints_outside_image(coco: COCO) -> None:
    """
    Check if there are any keypoints outside the image.
    :param coco:
    :return: None
    """
    for ann in coco.anns.values():
        keypoints = np.array(ann["keypoints"]).reshape(-1, 3)
        image_rows = coco.imgs[ann["image_id"]]["height"]
        image_cols = coco.imgs[ann["image_id"]]["width"]

        visible_joints = keypoints[:, 2] > 0
        joints_outside_image = (keypoints[:, 0] < 0) | (keypoints[:, 0] >= image_cols) | (keypoints[:, 1] < 0) | (keypoints[:, 1] >= image_rows)
        visible_joints_outside_image = visible_joints & joints_outside_image
        if visible_joints_outside_image.any():
            logger.warning(
                f"Annotation {ann['id']} for image {ann['image_id']} (width={image_cols}, height={image_rows}) "
                f"contains keypoints outside image boundary {keypoints[joints_outside_image]}. "
            )


def check_for_duplicate_annotations(coco: COCO, max_distance_threshold=2) -> None:
    """
    Check if there are any duplicate (overlapping) object annotations.
    :param coco:
    :param max_distance_threshold: Maximum average distance between keypoints of two instances to be considered as duplicate.
    :return: None
    """

    image_ids = list(coco.imgs.keys())
    for image_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(ann_ids)

        joints = []
        for ann in annotations:
            keypoints = np.array(ann["keypoints"]).reshape(-1, 3)
            joints.append(keypoints[:, :2])

        if len(joints) == 0:
            continue

        gt_joints1 = np.expand_dims(joints, axis=0)  # [1, Num_people, Num_joints, 2]
        gt_joints2 = np.expand_dims(joints, axis=1)  # [Num_people, 1, Num_joints, 2]
        diff = np.sqrt(np.sum((gt_joints1 - gt_joints2) ** 2, axis=-1))  # [Num_people, Num_people, Num_joints]
        diffmean = np.mean(diff, axis=-1)

        duplicate_mask = np.triu(diffmean < max_distance_threshold, k=1)
        duplicate_indexes_i, duplicate_indexes_j = np.nonzero(duplicate_mask)

        for i, j in zip(duplicate_indexes_i, duplicate_indexes_j):
            logger.warning(f"Duplicate annotations for image {image_id}: {annotations[i]['id']} and {annotations[j]['id']}")


def make_keypoints_outside_image_invisible(coco: COCO) -> COCO:
    for ann in coco.anns.values():
        keypoints = np.array(ann["keypoints"]).reshape(-1, 3)
        image_rows = coco.imgs[ann["image_id"]]["height"]
        image_cols = coco.imgs[ann["image_id"]]["width"]

        visible_joints = keypoints[:, 2] > 0
        joints_outside_image = (keypoints[:, 0] < 0) | (keypoints[:, 0] >= image_cols) | (keypoints[:, 1] < 0) | (keypoints[:, 1] >= image_rows)
        visible_joints_outside_image = visible_joints & joints_outside_image
        if visible_joints_outside_image.any():
            logger.debug(
                f"Detected GT joints outside image size (width={image_cols}, height={image_rows}). "
                f"{keypoints[joints_outside_image]} for obj_id {ann['id']} image_id {ann['image_id']}. "
                f"Changing visibility to invisible."
            )
            keypoints[visible_joints_outside_image, 2] = 0

            ann["keypoints"] = [float(x) for x in keypoints.reshape(-1)]
    return coco


def remove_duplicate_annotations(coco: COCO) -> COCO:
    ann_to_remove = []

    image_ids = list(coco.imgs.keys())
    for image_id in tqdm(image_ids, desc="Removing duplicate annotations"):
        ann_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(ann_ids)

        joints = []
        for ann in annotations:
            keypoints = np.array(ann["keypoints"]).reshape(-1, 3)
            joints.append(keypoints[:, :2])

        if len(joints) == 0:
            continue

        joints = np.stack(joints, axis=0)
        gt_joints1 = np.expand_dims(joints, axis=0)  # [1, Num_people, Num_joints, 2]
        gt_joints2 = np.expand_dims(joints, axis=1)  # [Num_people, 1, Num_joints, 2]
        diff = np.sqrt(np.sum((gt_joints1 - gt_joints2) ** 2, axis=-1))  # [Num_people, Num_people, Num_joints]
        diffmean = np.mean(diff, axis=-1)

        duplicate_mask = np.triu(diffmean < 2, k=1)
        duplicate_indexes_i, duplicate_indexes_j = np.nonzero(duplicate_mask)

        for j in duplicate_indexes_j:
            ann_to_remove.append(ann_ids[j])

    if len(ann_to_remove) > 0:
        ann_to_remove = set(ann_to_remove)
        logger.debug(f"Removing {len(ann_to_remove)} duplicate annotations")
        len_before = len(coco.dataset["annotations"])
        coco.dataset["annotations"] = [v for v in coco.dataset["annotations"] if v["id"] not in ann_to_remove]
        len_after = len(coco.dataset["annotations"])
        logger.debug(f"Removed {len_before - len_after} duplicate annotations")
        coco.createIndex()

    return coco


def remove_crowd_annotations(coco: COCO):
    ann_to_remove = []

    image_ids = list(coco.imgs.keys())
    for image_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(ann_ids)

        for ann in annotations:
            if bool(ann["iscrowd"]):
                ann_to_remove.append(ann["id"])

    if len(ann_to_remove) > 0:
        logger.debug(f"Removing {len(ann_to_remove)} crowd annotations")
        len_before = len(coco.dataset["annotations"])
        coco.dataset["annotations"] = [v for v in coco.dataset["annotations"] if v["id"] not in ann_to_remove]
        len_after = len(coco.dataset["annotations"])
        logger.debug(f"Removed {len_before - len_after} crowd annotations")
        coco.createIndex()

    return coco


def remove_empty_samples(coco: COCO):
    img_ids_to_remove = []

    image_ids = list(coco.imgs.keys())
    for image_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(ann_ids)
        if len(annotations) == 0:
            img_ids_to_remove.append(image_id)

    if len(img_ids_to_remove) > 0:
        logger.debug(f"Removing {len(img_ids_to_remove)} empty images")
        len_before = len(coco.dataset["images"])
        coco.dataset["images"] = [v for v in coco.dataset["images"] if v["id"] not in img_ids_to_remove]
        len_after = len(coco.dataset["images"])
        logger.debug(f"Removed {len_before - len_after} empty images")
        coco.createIndex()

    return coco


def remove_samples_with_crowd_annotations(coco: COCO):
    img_ids_to_remove = []

    image_ids = list(coco.imgs.keys())
    for image_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(ann_ids)

        for ann in annotations:
            if bool(ann["iscrowd"]):
                img_ids_to_remove.append(image_id)
                break

    if len(img_ids_to_remove) > 0:
        logger.debug(f"Removing {len(img_ids_to_remove)} empty images")
        len_before = len(coco.dataset["images"])
        coco.dataset["images"] = [v for v in coco.dataset["images"] if v["id"] not in img_ids_to_remove]
        len_after = len(coco.dataset["images"])
        logger.debug(f"Removed {len_before - len_after} empty images")
        coco.createIndex()

    return coco


@dataclasses.dataclass
class KeypointsAnnotation:
    image_path: str
    image_width: int
    image_height: int

    ann_boxes_xyxy: np.ndarray
    ann_is_crowd: np.ndarray
    ann_area: np.ndarray
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

    if len(coco.dataset["categories"]) != 1:
        raise ValueError("Dataset must contain exactly one category")

    # Extract class names and class ids
    category_name = coco["categories"][0]["name"]
    keypoints = coco.dataset["categories"][0]["keypoints"]
    num_keypoints = len(keypoints)

    # Extract box annotations
    ann_box_xyxy = xywh_to_xyxy_inplace(np.array([annotation["bbox"] for annotation in coco["annotations"]], dtype=np.float32), image_shape=None)
    ann_keypoints = np.stack([np.array(annotation["keypoints"], dtype=np.float32).reshape(num_keypoints, 3) for annotation in coco["annotations"]])
    ann_iscrowd = np.array([annotation["iscrowd"] for annotation in coco["annotations"]], dtype=bool)
    ann_image_ids = np.array([annotation["image_id"] for annotation in coco["annotations"]], dtype=int)
    ann_segmentations = np.array([annotation["segmentation"] for annotation in coco["annotations"]], dtype=str)

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
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            ann_boxes_xyxy=ann_box_xyxy[mask],
            ann_is_crowd=ann_iscrowd[mask],
            ann_area=ann_areas[mask],
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
            ann.ann_area = ann.ann_area[keep_mask]
            ann.ann_segmentations = ann.ann_segmentations[keep_mask]
            ann.ann_is_crowd = ann.ann_is_crowd[keep_mask]

        if crowd_annotations_action == CrowdAnnotationActionEnum.DROP_SAMPLE:
            if ann.ann_is_crowd.any():
                continue

        annotations.append(ann)

    return category_name, keypoints, annotations


def rle2mask(rle: str, image_shape: Tuple[int, int]):
    """
    Convert RLE to binary mask
    :param rle: A string containing RLE-encoded mask
    :param image_shape: Output image shape (rows, cols)
    :return: A decoded binary mask
    """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(image_shape[0] * image_shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(image_shape)
