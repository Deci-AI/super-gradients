import os
from typing import Tuple, List, Mapping, Any

import cv2
import numpy as np
import pycocotools
import torch
from pycocotools.coco import COCO
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.target_generator_factory import TargetGeneratorsFactory
from super_gradients.common.factories.transforms_factory import TransformsFactory
from torch.utils.data import default_collate, Dataset

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.transforms.keypoint_transforms import KeypointsCompose

logger = get_logger(__name__)


class COCOKeypointsDataset(Dataset):
    """
    Dataset class for training pose estimation models on COCO Keypoints dataset.
    Use should pass a target generator class that is model-specific and generates the targets for the model.
    """

    @resolve_param("transforms", TransformsFactory())
    @resolve_param("target_generator", TargetGeneratorsFactory())
    def __init__(
        self,
        root_dir: str,
        images_dir: str,
        json_file: str,
        include_empty_samples: bool,
        target_generator,
        transforms: KeypointsCompose,
        min_instance_area: float,
    ):
        """

        :param root_dir: Root directory of the COCO dataset
        :param images_dir: path suffix to the images directory inside the dataset_root
        :param json_file: path suffix to the json file inside the dataset_root
        :param include_empty_samples: if True, images without any annotations will be included in the dataset.
            Otherwise, they will be filtered out.
        :param target_generator: Target generator that will be used to generate the targets for the model.
            See DEKRTargetsGenerator for an example.
        :param transforms: Transforms to be applied to the image & keypoints
        :param min_instance_area: Minimum area of an instance to be included in the dataset
        """
        super().__init__()
        self.root = root_dir
        self.images_dir = os.path.join(root_dir, images_dir)
        self.json_file = os.path.join(root_dir, json_file)

        coco = COCO(self.json_file)
        if len(coco.dataset["categories"]) != 1:
            raise ValueError("Dataset must contain exactly one category")

        self.coco = coco
        self.ids = list(self.coco.imgs.keys())
        self.joints = coco.dataset["categories"][0]["keypoints"]
        self.num_joints = len(self.joints)
        self.min_instance_area = min_instance_area

        self.transforms = transforms
        self.target_generator = target_generator

        if not include_empty_samples:
            subset = [img_id for img_id in self.ids if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0]
            self.ids = subset

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any, Mapping[str, Any]]:
        coco = self.coco
        img_id = self.ids[index]
        image_info = coco.loadImgs(img_id)[0]
        file_name = image_info["file_name"]
        file_path = os.path.join(self.images_dir, file_name)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)
        anno = [obj for obj in anno if bool(obj["iscrowd"]) is False and obj["num_keypoints"] > 0]

        orig_image = cv2.imread(file_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if orig_image.shape[0] != image_info["height"] or orig_image.shape[1] != image_info["width"]:
            raise RuntimeError(f"Annotated image size ({image_info['height'],image_info['width']}) does not match image size in file {orig_image.shape[:2]}")

        joints: np.ndarray = self.get_joints(anno)
        mask: np.ndarray = self.get_mask(anno, image_info)

        img, mask, joints = self.transforms(orig_image, mask, joints)

        joints = self.filter_joints(joints, img)

        targets = self.target_generator(img, joints, mask)
        return img, targets, {"joints": joints, "file_name": image_info["file_name"]}

    def compute_area(self, joints: np.ndarray) -> np.ndarray:
        """
        Compute area of a bounding box for each instance.
        :param joints:  [Num Instances, Num Joints, 3]
        :return: [Num Instances]
        """
        w = np.max(joints[:, :, 0], axis=-1) - np.min(joints[:, :, 0], axis=-1)
        h = np.max(joints[:, :, 1], axis=-1) - np.min(joints[:, :, 1], axis=-1)
        return w * h

    def filter_joints(self, joints: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Filter instances that are either too small or do not have visible keypoints
        :param joints: Array of shape [Num Instances, Num Joints, 3]
        :param image:
        :return: [New Num Instances, Num Joints, 3], New Num Instances <= Num Instances
        """
        # Update visibility of joints for those that are outside the image
        outside_image_mask = (joints[:, :, 0] < 0) | (joints[:, :, 1] < 0) | (joints[:, :, 0] >= image.shape[1]) | (joints[:, :, 1] >= image.shape[0])
        joints[outside_image_mask, 2] = 0

        # Filter instances with all invisible keypoints
        instances_with_visible_joints = np.count_nonzero(joints[:, :, 2], axis=-1) > 0
        joints = joints[instances_with_visible_joints]

        # Remove instances with too small area
        areas = self.compute_area(joints)
        joints = joints[areas > self.min_instance_area]

        return joints

    def get_joints(self, anno: List[Mapping[str, Any]]) -> np.ndarray:
        """
        Decode the keypoints from the COCO annotation and return them as an array of shape [Num Instances, Num Joints, 3].
        The visibility of keypoints is encoded in the third dimension of the array with following values:
         - 0 being invisible (outside image)
         - 1 present in image but occluded
         - 2 - fully visible
        :param anno:
        :return: [Num Instances, Num Joints, 3], where last channel represents (x, y, visibility)
        """
        joints = []

        for i, obj in enumerate(anno):
            keypoints = np.array(obj["keypoints"]).reshape([-1, 3])
            joints.append(keypoints)

        num_instances = len(joints)
        joints = np.array(joints, dtype=np.float32).reshape((num_instances, self.num_joints, 3))
        return joints

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
    Collate image & targets, return extras as is.
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
