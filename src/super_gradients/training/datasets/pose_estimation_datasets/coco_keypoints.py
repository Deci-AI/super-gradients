from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Tuple, List

import cv2
import numpy as np
import pycocotools
import torch
from pycocotools.coco import COCO
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.pose_estimation_datasets.target_generators import HeatmapGenerator, OffsetGenerator
from super_gradients.training.transforms.keypoint_transforms import KeypointTransform
from torch.utils.data import default_collate, Dataset

logger = get_logger(__name__)


class COCOKeypoints(Dataset):
    """ """

    def __init__(
        self,
        dataset_root: str,
        images_dir: str,
        json_file: str,
        num_joints: int,
        targets_size: Tuple[int, int],
        offset_radius: float,
        sigma: float,
        center_sigma: float,
        bg_weight: float,
        include_empty_samples: bool,
        include_crowd_targets: bool,
        transforms: List[KeypointTransform],
        min_object_area: float = 32**2,
    ):

        self.root = dataset_root
        self.images_dir = os.path.join(dataset_root, images_dir)
        self.json_file = os.path.join(dataset_root, json_file)

        coco = COCO(self.json_file)
        self.coco = coco
        self.ids = list(self.coco.imgs.keys())
        self.num_joints = num_joints

        cats = [cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ["__background__"] + cats
        logger.info("=> classes: {}".format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls], self._class_to_ind[cls]) for cls in self.classes[1:]])

        self.num_joints_with_center = self.num_joints + 1
        self.min_object_area = min_object_area
        self.include_crowd_targets = include_crowd_targets

        self.heatmap_generator = HeatmapGenerator(
            output_res=targets_size,
            num_joints=num_joints,
            sigma=sigma,
            center_sigma=center_sigma,
            bg_weight=bg_weight,
        )
        self.offset_generator = OffsetGenerator(
            output_res=targets_size,
            num_joints=num_joints,
            radius=offset_radius,
        )
        self.transforms = transforms

        if not include_empty_samples:
            subset = [img_id for img_id in self.ids if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0]
            self.ids = subset

    def _get_image_path(self, file_name):
        return os.path.join(self.images_dir, file_name)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)
        image_info = coco.loadImgs(img_id)[0]
        file_name = image_info["file_name"]

        orig_image = cv2.imread(self._get_image_path(file_name), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

        mask: np.ndarray = self.get_mask(anno, image_info)
        if orig_image.shape[0] != image_info["height"] or orig_image.shape[1] != image_info["width"]:
            raise RuntimeError(f"Annotated image size ({image_info['height'],image_info['width']}) does not match image size in file {orig_image.shape[:2]}")

        if self.include_crowd_targets:
            anno = [obj for obj in anno if obj["num_keypoints"] > 0]
        else:
            anno = [obj for obj in anno if bool(obj["iscrowd"]) is False and obj["num_keypoints"] > 0]

        original_joints, area = self.get_joints(anno, image_shape=orig_image.shape[:2])
        pose_scale_factor = 1.0

        img, [mask], [joints], area, pose_scale_factor = self.transforms(orig_image, [mask], [original_joints], area, pose_scale_factor)

        heatmap, mask, joints = self.heatmap_generator(joints, mask)
        offset, offset_weight = self.offset_generator(joints, area)

        targets = (heatmap, mask, offset, offset_weight)
        return img, targets, {"joints": joints, "area": area, "anno": anno, "file_name": image_info["file_name"], "pose_scale_factor": pose_scale_factor}

    def cal_area_2_torch(self, v):
        w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
        h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
        return w * w + h * h

    def get_joints(self, anno, image_shape: Tuple[int, int]):
        image_rows, image_cols = image_shape
        joints = []
        areas = []

        for i, obj in enumerate(anno):
            keypoints = np.array(obj["keypoints"]).reshape([-1, 3])

            area = self.cal_area_2_torch(torch.tensor(keypoints[None, ...]))

            if obj["area"] < self.min_object_area:
                continue

            # Computing a center point for each person
            visible_keypoints = keypoints[:, 2] > 0
            joints_sum = np.sum(keypoints[:, :2] * np.expand_dims(visible_keypoints, -1), axis=0)
            num_vis_joints = np.count_nonzero(visible_keypoints)
            if num_vis_joints <= 0:
                continue

            keypoints_with_center = np.zeros((self.num_joints_with_center, 3))
            keypoints_with_center[0 : self.num_joints] = keypoints
            keypoints_with_center[-1, :2] = joints_sum / num_vis_joints
            keypoints_with_center[-1, 2] = 1

            joints.append(keypoints_with_center)
            areas.append(float(area))

        areas = np.array(areas, dtype=np.float32).reshape((-1, 1))
        joints = np.array(joints, dtype=np.float32).reshape((-1, self.num_joints_with_center, 3))
        return joints, areas

    def get_mask(self, anno, img_info) -> np.ndarray:
        """
        This method computes ignore mask, which describes crowd objects / objects w/o keypoints to exclude these predictions from contributing to the loss
        :param anno:
        :param img_info:
        :return: Float mask of [H,W] shape (same as image dimensions),
            where 1.0 values corresponds to pixels that should contribute to the loss, and 0.0 pixels indicates areas that should be excluded.
        """
        m = np.zeros((img_info["height"], img_info["width"]), dtype=np.float32)

        for obj in anno:
            if obj["iscrowd"]:
                rle = pycocotools.mask.frPyObjects(obj["segmentation"], img_info["height"], img_info["width"])
                mask = pycocotools.mask.decode(rle)
                if mask.shape != m.shape:
                    logger.warning(f"Mask shape {mask.shape} does not match image shape {m.shape} for image {img_info['file_name']}")
                    continue
                m += mask
            elif obj["num_keypoints"] == 0:
                rles = pycocotools.mask.frPyObjects(obj["segmentation"], img_info["height"], img_info["width"])
                for rle in rles:
                    mask = pycocotools.mask.decode(rle)
                    if mask.shape != m.shape:
                        logger.warning(f"Mask shape {mask.shape} does not match image shape {m.shape} for image {img_info['file_name']}")
                        continue

                    m += mask

        return (m < 0.5).astype(np.float32)


class COCOKeypointsCollate:
    """
    Collate image & targets, return extras as is
    """

    def __call__(self, batch):
        images = []
        targets = []
        extras = []
        for image, target, extra in batch:
            images.append(image)
            targets.append(target)
            extras.append(extra)

        extras = {k: [dic[k] for dic in extras] for k in extras[0]}  # Convert list of dicts to dict of lists

        images = default_collate(images)
        targets = default_collate(targets)
        return images, targets, extras
