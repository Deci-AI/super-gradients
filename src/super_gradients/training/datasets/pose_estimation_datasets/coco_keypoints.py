import os
from typing import Tuple, List, Mapping, Any, Union

import cv2
import numpy as np
import pycocotools
from pycocotools.coco import COCO
from torch import Tensor

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.object_names import Datasets, Processings
from super_gradients.common.registry.registry import register_dataset
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.target_generator_factory import TargetGeneratorsFactory
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training.datasets.pose_estimation_datasets.base_keypoints import BaseKeypointsDataset
from super_gradients.training.transforms.keypoint_transforms import KeypointTransform

logger = get_logger(__name__)


@register_dataset(Datasets.COCO_KEY_POINTS_DATASET)
class COCOKeypointsDataset(BaseKeypointsDataset):
    """
    Dataset class for training pose estimation models on COCO Keypoints dataset.
    Use should pass a target generator class that is model-specific and generates the targets for the model.
    """

    @resolve_param("transforms", TransformsFactory())
    @resolve_param("target_generator", TargetGeneratorsFactory())
    def __init__(
        self,
        data_dir: str,
        images_dir: str,
        json_file: str,
        include_empty_samples: bool,
        target_generator,
        transforms: List[KeypointTransform],
        min_instance_area: float,
        edge_links: Union[List[Tuple[int, int]], np.ndarray],
        edge_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
        keypoint_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
    ):
        """

        :param data_dir: Root directory of the COCO dataset
        :param images_dir: path suffix to the images directory inside the dataset_root
        :param json_file: path suffix to the json file inside the dataset_root
        :param include_empty_samples: if True, images without any annotations will be included in the dataset.
            Otherwise, they will be filtered out.
        :param target_generator: Target generator that will be used to generate the targets for the model.
            See DEKRTargetsGenerator for an example.
        :param transforms: Transforms to be applied to the image & keypoints
        :param min_instance_area: Minimum area of an instance to be included in the dataset
        :param edge_links: Edge links between joints
        :param edge_colors: Color of the edge links. If None, the color will be generated randomly.
        :param keypoint_colors: Color of the keypoints. If None, the color will be generated randomly.
        """

        json_file = os.path.join(data_dir, json_file)
        coco = COCO(json_file)
        if len(coco.dataset["categories"]) != 1:
            raise ValueError("Dataset must contain exactly one category")
        joints = coco.dataset["categories"][0]["keypoints"]
        num_joints = len(joints)

        super().__init__(
            transforms=transforms,
            target_generator=target_generator,
            min_instance_area=min_instance_area,
            num_joints=num_joints,
            edge_links=edge_links,
            edge_colors=edge_colors,
            keypoint_colors=keypoint_colors,
        )
        self.root = data_dir
        self.images_dir = os.path.join(data_dir, images_dir)
        self.coco = coco
        self.ids = list(self.coco.imgs.keys())
        self.joints = joints

        if not include_empty_samples:
            subset = [img_id for img_id in self.ids if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0]
            self.ids = subset

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[Tensor, Any, Mapping[str, Any]]:
        img, mask, gt_joints, gt_areas, gt_bboxes, gt_iscrowd = self.load_sample(index)
        img, mask, gt_joints, gt_areas, gt_bboxes = self.transforms(img, mask, gt_joints, areas=gt_areas, bboxes=gt_bboxes)

        image_shape = img.size(1), img.size(2)
        gt_joints, gt_areas, gt_bboxes, gt_iscrowd = self.filter_joints(image_shape, gt_joints, gt_areas, gt_bboxes, gt_iscrowd)

        targets = self.target_generator(img, gt_joints, mask)
        return img, targets, {"gt_joints": gt_joints, "gt_bboxes": gt_bboxes, "gt_iscrowd": gt_iscrowd, "gt_areas": gt_areas}

    def load_sample(self, index):
        img_id = self.ids[index]
        image_info = self.coco.loadImgs(img_id)[0]
        file_name = image_info["file_name"]
        file_path = os.path.join(self.images_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)

        gt_iscrowd = np.array([bool(ann["iscrowd"]) for ann in anno]).reshape((-1))
        gt_bboxes = np.array([ann["bbox"] for ann in anno], dtype=np.float32).reshape((-1, 4))
        gt_areas = np.array([ann["area"] for ann in anno], dtype=np.float32).reshape((-1))

        orig_image = cv2.imread(file_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if orig_image.shape[0] != image_info["height"] or orig_image.shape[1] != image_info["width"]:
            raise RuntimeError(f"Annotated image size ({image_info['height'],image_info['width']}) does not match image size in file {orig_image.shape[:2]}")

        joints: np.ndarray = self.get_joints(anno)
        mask: np.ndarray = self.get_mask(anno, image_info)

        return orig_image, mask, joints, gt_areas, gt_bboxes, gt_iscrowd

    def filter_joints(
        self,
        image_shape,
        joints: np.ndarray,
        areas: np.ndarray,
        bboxes: np.ndarray,
        is_crowd: np.ndarray,
    ):
        """
        Filter instances that are either too small or do not have visible keypoints.

        :param image: Image if [H,W,C] shape. Used to infer image boundaries
        :param joints: Array of shape [Num Instances, Num Joints, 3]
        :param areas: Array of shape [Num Instances] with area of each instance.
                      Instance area comes from segmentation mask from COCO annotation file.
        :param bboxes: Array of shape [Num Instances, 4] for bounding boxes in XYWH format.
                       Bounding boxes comes from segmentation mask from COCO annotation file.
        :param: is_crowd: Array of shape [Num Instances] indicating whether an instance is a crowd target.
        :return: [New Num Instances, Num Joints, 3], New Num Instances <= Num Instances
        """

        # Update visibility of joints for those that are outside the image
        outside_image_mask = (joints[:, :, 0] < 0) | (joints[:, :, 1] < 0) | (joints[:, :, 0] >= image_shape[1]) | (joints[:, :, 1] >= image_shape[0])
        joints[outside_image_mask, 2] = 0

        # Filter instances with all invisible keypoints
        instances_with_visible_joints = np.count_nonzero(joints[:, :, 2], axis=-1) > 0
        instances_with_good_area = areas > self.min_instance_area

        keep_mask = instances_with_visible_joints & instances_with_good_area

        joints = joints[keep_mask]
        areas = areas[keep_mask]
        bboxes = bboxes[keep_mask]
        is_crowd = is_crowd[keep_mask]

        return joints, areas, bboxes, is_crowd

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

    def get_dataset_preprocessing_params(self):
        """

        :return:
        """
        # Since we are using cv2.imread to read images, our model in fact is trained on BGR images.
        # In our pipelines the convention that input images are RGB, so we need to reverse the channels to get BGR
        # to match with the expected input of the model.
        pipeline = [Processings.ReverseImageChannels] + self.transforms.get_equivalent_preprocessing()
        params = dict(
            conf=0.25,
            image_processor={Processings.ComposeProcessing: {"processings": pipeline}},
            edge_links=self.edge_links,
            edge_colors=self.edge_colors,
            keypoint_colors=self.keypoint_colors,
        )
        return params
