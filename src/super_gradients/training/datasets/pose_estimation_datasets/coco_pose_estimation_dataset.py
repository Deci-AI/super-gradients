import os
from typing import Tuple, List, Union

import cv2
import numpy as np
import pycocotools
from pycocotools.coco import COCO

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.common.object_names import Datasets, Processings
from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy, xyxy_to_xywh
from super_gradients.training.datasets.pose_estimation_datasets.abstract_pose_estimation_dataset import AbstractPoseEstimationDataset
from super_gradients.training.datasets.pose_estimation_datasets.coco_utils import (
    CrowdAnnotationActionEnum,
    remove_duplicate_annotations as remove_duplicate_annotations_fn,
    remove_crowd_annotations,
    remove_samples_with_crowd_annotations,
)
from super_gradients.training.samples import PoseEstimationSample
from super_gradients.training.transforms.keypoint_transforms import AbstractKeypointTransform

logger = get_logger(__name__)


@register_dataset(Datasets.COCO_POSE_ESTIMATION_DATASET)
class COCOPoseEstimationDataset(AbstractPoseEstimationDataset):
    """
    Dataset class for training pose estimation models on COCO dataset.
    Use should pass a target generator class that is model-specific and generates the targets for the model.
    """

    @resolve_param("transforms", TransformsFactory())
    def __init__(
        self,
        data_dir: str,
        images_dir: str,
        json_file: str,
        include_empty_samples: bool,
        transforms: List[AbstractKeypointTransform],
        edge_links: Union[List[Tuple[int, int]], np.ndarray],
        edge_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
        keypoint_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
        remove_duplicate_annotations: bool = False,
        crowd_annotations_action: Union[str, CrowdAnnotationActionEnum] = CrowdAnnotationActionEnum.NO_ACTION,
    ):
        """

        :param data_dir: Root directory of the COCO dataset
        :param images_dir: path suffix to the images directory inside the dataset_root
        :param json_file: path suffix to the json file inside the dataset_root
        :param include_empty_samples: if True, images without any annotations will be included in the dataset.
            Otherwise, they will be filtered out.
        :param transforms: Transforms to be applied to the image & keypoints
        :param edge_links: Edge links between joints
        :param edge_colors: Color of the edge links. If None, the color will be generated randomly.
        :param keypoint_colors: Color of the keypoints. If None, the color will be generated randomly.
        :param crowd_annotations_action: Action to take for annotations with iscrowd=1. Can be one of the following:
            - "drop_sample" - Samples with crowd annotations will be dropped from the dataset.
            - "drop_annotation" - Crowd annotations will be dropped from the dataset.
            - "mask_as_normal" - These annotations will be treated as normal (non-crowd) annotations.
            - "no_action" - No action will be taken for crowd annotations.
        """

        crowd_annotations_action = CrowdAnnotationActionEnum(crowd_annotations_action)
        if crowd_annotations_action not in [
            CrowdAnnotationActionEnum.NO_ACTION,
            CrowdAnnotationActionEnum.DROP_ANNOTATION,
            CrowdAnnotationActionEnum.DROP_SAMPLE,
            CrowdAnnotationActionEnum.MASK_AS_NORMAL,
        ]:
            raise ValueError(f"crowd_annotations_action must be one of CrowdAnnotationActionEnum values, got {crowd_annotations_action}")

        json_file = os.path.join(data_dir, json_file)
        if not os.path.exists(json_file) or not os.path.isfile(json_file):
            raise FileNotFoundError(f"Annotation file {json_file} does not exist")

        coco = COCO(json_file)

        if remove_duplicate_annotations:
            coco = remove_duplicate_annotations_fn(coco)

        if crowd_annotations_action == CrowdAnnotationActionEnum.DROP_SAMPLE:
            coco = remove_samples_with_crowd_annotations(coco)
        elif crowd_annotations_action == CrowdAnnotationActionEnum.DROP_ANNOTATION:
            coco = remove_crowd_annotations(coco)

        if len(coco.dataset["categories"]) != 1:
            raise ValueError("Dataset must contain exactly one category")
        joints = coco.dataset["categories"][0]["keypoints"]
        num_joints = len(joints)

        super().__init__(
            transforms=transforms,
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
        self.crowd_annotations_action = crowd_annotations_action

        if not include_empty_samples:
            subset = [img_id for img_id in self.ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
            self.ids = subset

    def __len__(self):
        return len(self.ids)

    def load_sample(self, index: int) -> PoseEstimationSample:
        """

        :param index:
        :return: Tuple of (image, mask, joints, instance areas, instance bounding boxes, is_crowd)
        """
        img_id = self.ids[index]
        image_info = self.coco.loadImgs(img_id)[0]
        file_name = image_info["file_name"]
        file_path = os.path.join(self.images_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)

        gt_iscrowd = np.array([bool(ann["iscrowd"]) for ann in anno]).reshape((-1))

        if self.crowd_annotations_action == CrowdAnnotationActionEnum.MASK_AS_NORMAL:
            # If crowd_annotations_action is "include", we treat crowd annotations as normal annotations
            # so we set is_crowd to False for all annotations
            gt_iscrowd = np.zeros_like(gt_iscrowd, dtype=bool)

        gt_joints = np.array([ann["keypoints"] for ann in anno], dtype=np.float32).reshape((-1, self.num_joints, 3))
        gt_bboxes = np.array([ann["bbox"] for ann in anno], dtype=np.float32).reshape((-1, 4))
        gt_areas = np.array([ann["area"] for ann in anno], dtype=np.float32).reshape((-1))

        orig_image = cv2.imread(file_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if orig_image.shape[0] != image_info["height"] or orig_image.shape[1] != image_info["width"]:
            raise RuntimeError(f"Annotated image size ({image_info['height'],image_info['width']}) does not match image size in file {orig_image.shape[:2]}")

        # clip bboxes (xywh) to image boundaries
        image_height, image_width = orig_image.shape[:2]
        xyxy_bboxes = xywh_to_xyxy(gt_bboxes, image_shape=(image_height, image_width))
        image_height, image_width = orig_image.shape[:2]
        xyxy_bboxes[:, 0] = np.clip(xyxy_bboxes[:, 0], 0, image_width)
        xyxy_bboxes[:, 1] = np.clip(xyxy_bboxes[:, 1], 0, image_height)
        xyxy_bboxes[:, 2] = np.clip(xyxy_bboxes[:, 2], 0, image_width)
        xyxy_bboxes[:, 3] = np.clip(xyxy_bboxes[:, 3], 0, image_height)
        gt_bboxes = xyxy_to_xywh(xyxy_bboxes, image_shape=(image_height, image_width))

        mask: np.ndarray = self.get_mask(anno, image_info)

        return PoseEstimationSample(
            image=orig_image, mask=mask, joints=gt_joints, areas=gt_areas, bboxes=gt_bboxes, is_crowd=gt_iscrowd, additional_samples=None
        )

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
            conf=0.05,
            image_processor={Processings.ComposeProcessing: {"processings": pipeline}},
            edge_links=self.edge_links,
            edge_colors=self.edge_colors,
            keypoint_colors=self.keypoint_colors,
        )
        return params