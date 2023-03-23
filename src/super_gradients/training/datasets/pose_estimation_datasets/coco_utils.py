import numpy as np
from pycocotools.coco import COCO

from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)

__all__ = ["check_keypoints_outside_image", "check_for_duplicate_annotations", "make_keypoints_outside_image_invisible", "remove_duplicate_annotations"]


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

        duplicate_mask = np.triu(diffmean < 2, k=1)
        duplicate_indexes_i, duplicate_indexes_j = np.nonzero(duplicate_mask)

        for j in duplicate_indexes_j:
            ann_to_remove.append(ann_ids[j])

    if len(ann_to_remove) > 0:
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
