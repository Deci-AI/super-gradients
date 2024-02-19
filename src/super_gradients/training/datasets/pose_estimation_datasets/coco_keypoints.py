import os
from typing import Tuple, List, Mapping, Any, Union

import cv2
import numpy as np
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.target_generator_factory import TargetGeneratorsFactory
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.common.object_names import Datasets, Processings
from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xyxy_to_xywh
from super_gradients.training.datasets.pose_estimation_datasets.base_keypoints import BaseKeypointsDataset
from super_gradients.training.datasets.pose_estimation_datasets.coco_utils import (
    parse_coco_into_keypoints_annotations,
    CrowdAnnotationActionEnum,
    segmentation2mask,
)
from super_gradients.training.transforms.keypoint_transforms import KeypointTransform
from torch import Tensor

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
        self.category_name, self.joints, self.annotations = parse_coco_into_keypoints_annotations(
            json_file,
            image_path_prefix=os.path.join(data_dir, images_dir),
            remove_duplicate_annotations=False,
            crowd_annotations_action=CrowdAnnotationActionEnum.NO_ACTION,
        )
        num_joints = len(self.joints)
        super().__init__(
            transforms=transforms,
            target_generator=target_generator,
            min_instance_area=min_instance_area,
            num_joints=num_joints,
            edge_links=edge_links,
            edge_colors=edge_colors,
            keypoint_colors=keypoint_colors,
        )

        self.non_empty_annotation_indexes = np.argwhere([len(ann.ann_keypoints) > 0 for ann in self.annotations]).flatten()
        self.include_empty_samples = include_empty_samples

    def __len__(self):
        if self.include_empty_samples:
            return len(self.annotations)
        else:
            return len(self.non_empty_annotation_indexes)

    def __getitem__(self, index: int) -> Tuple[Tensor, Any, Mapping[str, Any]]:
        img, mask, gt_joints, gt_areas, gt_bboxes, gt_iscrowd = self.load_sample(index)
        img, mask, gt_joints, gt_areas, gt_bboxes = self.transforms(img, mask, gt_joints, areas=gt_areas, bboxes=gt_bboxes)

        image_shape = img.size(1), img.size(2)
        gt_joints, gt_areas, gt_bboxes, gt_iscrowd = self.filter_joints(image_shape, gt_joints, gt_areas, gt_bboxes, gt_iscrowd)

        targets = self.target_generator(img, gt_joints, mask)
        return img, targets, {"gt_joints": gt_joints, "gt_bboxes": gt_bboxes, "gt_iscrowd": gt_iscrowd, "gt_areas": gt_areas}

    def load_sample(self, index):
        if not self.include_empty_samples:
            index = self.non_empty_annotation_indexes[index]
        ann = self.annotations[index]

        image_shape = (ann.image_height, ann.image_width)

        gt_iscrowd = ann.ann_is_crowd.copy()
        gt_joints = ann.ann_keypoints.copy()
        gt_bboxes = ann.ann_boxes_xyxy.copy()
        gt_segmentations = ann.ann_segmentations
        gt_areas = ann.ann_areas.copy()

        orig_image = cv2.imread(ann.image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if orig_image is None:
            # This is a nice fallback/hack to handle case when OpenCV cannot read some images
            # In happens to some OpenCV versions for COCO datasets (There are 1-2 corrupted images)
            # But we generaly want to read with OpenCV since it's much faster than PIL
            from PIL import Image

            orig_image = Image.open(ann.image_path).convert("BGR")

        if orig_image.shape[0] != ann.image_height or orig_image.shape[1] != ann.image_width:
            raise RuntimeError(f"Annotated image size ({ann.image_height,ann.image_width}) does not match image size in file {orig_image.shape[:2]}")

        # Clip bboxes to image boundaries (Some annotations extend 1-2px outside of image boundaries)
        image_height, image_width = orig_image.shape[:2]
        gt_bboxes[:, 0] = np.clip(gt_bboxes[:, 0], 0, image_width)
        gt_bboxes[:, 1] = np.clip(gt_bboxes[:, 1], 0, image_height)
        gt_bboxes[:, 2] = np.clip(gt_bboxes[:, 2], 0, image_width)
        gt_bboxes[:, 3] = np.clip(gt_bboxes[:, 3], 0, image_height)
        gt_bboxes_xywh = xyxy_to_xywh(gt_bboxes, image_shape=(image_height, image_width))

        mask: np.ndarray = self._get_crowd_mask(gt_segmentations[gt_iscrowd], image_shape)

        return orig_image, mask, gt_joints, gt_areas, gt_bboxes_xywh, gt_iscrowd

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

    def _get_crowd_mask(self, segmentations: List[str], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        This method computes ignore mask, which describes crowd objects / objects w/o keypoints to exclude these predictions from contributing to the loss
        :return: Float mask of [H,W] shape (same as image dimensions),
            where 1.0 values corresponds to pixels that should contribute to the loss, and 0.0 pixels indicates areas that should be excluded.
        """
        m = np.zeros(image_shape, dtype=bool)

        for segmentation in segmentations:
            mask = segmentation2mask(segmentation, image_shape)
            m[mask] = True

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
