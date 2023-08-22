from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import default_collate

from super_gradients.common.registry.registry import register_target_generator, register_collate_function
from .target_generators import KeypointsTargetsGenerator

__all__ = ["YoloNASPoseTargetsGenerator", "YoloNASPoseTargetsCollateFN"]

from ..data_formats.bbox_formats.xywh import xywh_to_xyxy


@register_target_generator()
class YoloNASPoseTargetsGenerator(KeypointsTargetsGenerator):
    """
    Target generator for YoloNASPose model.
    """

    def __init__(self):
        pass

    def __call__(self, image: Tensor, joints: np.ndarray, mask: np.ndarray, bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode the keypoints into dense targets that participate in loss computation.
        :param image: Image tensor [3, H, W]
        :param bboxes: Corresponding bounding boxes [NumInstances, 4] in (x, y, w, h) format
        :param joints: [Instances, NumJoints, 3]
        :param mask: [H,W] A mask that indicates which pixels should be included (1) or which one should be excluded (0) from loss computation.
        :return: A tuple of (boxes, joints, mask)
        """
        if image.shape[1:3] != mask.shape[:2]:
            raise ValueError(f"Image and mask should have the same shape {image.shape[1:3]} != {mask.shape[:2]}")

        boxes_xyxy = xywh_to_xyxy(bboxes, image_shape=None)
        return boxes_xyxy, joints


@register_collate_function()
class YoloNASPoseTargetsCollateFN:
    def __call__(self, batch):
        """
        Collate samples into a batch. This collate function should be used in conjunction with YoloNASPoseTargetsGenerator.

        :param batch: A list of samples from the dataset. Each sample is a tuple of (image, (boxes, joints), extras)
        :return: Tuple of (images, (boxes, joints), extras)
        - images: [Batch, 3, H, W]
        - boxes: [NumInstances, 5], last dimension represents (batch_index, x1, y1, x2, y2) of all boxes in a batch
        - joints: [NumInstances, NumJoints, 4] of all poses in a batch. Last dimension represents (batch_index, x, y, visibility)
        - extras: A dict of extra information per image need for metric computation
        """
        images = []
        all_boxes = []
        all_joints = []
        extras = []

        for image, (boxes, joints), extra in batch:
            images.append(image)
            all_boxes.append(torch.from_numpy(boxes))
            all_joints.append(torch.from_numpy(joints))
            extras.append(extra)

        images = default_collate(images)
        boxes = flat_collate_tensors_with_batch_index(all_boxes)
        joints = flat_collate_tensors_with_batch_index(all_joints)
        extras = {k: [dic[k] for dic in extras] for k in extras[0]}  # Convert list of dicts to dict of lists
        return images, (boxes, joints), extras


def flat_collate_tensors_with_batch_index(labels_batch: List[Tensor]) -> Tensor:
    """
    Stack a batch id column to targets and concatenate
    :param labels_batch: a list of targets per image (each of arbitrary length: [N1, ..., C], [N2, ..., C], [N3, ..., C],...)
    :return: A single tensor of shape [N1+N2+N3+..., ..., C+1], where N is the total number of targets in a batch
             and the 1st column is batch item index
    """
    labels_batch_indexed = []
    for i, labels in enumerate(labels_batch):
        batch_column = labels.new_ones(labels.shape[:-1] + (1,)) * i
        labels = torch.cat((batch_column, labels), dim=-1)
        labels_batch_indexed.append(labels)
    return torch.cat(labels_batch_indexed, 0)


def undo_flat_collate_tensors_with_batch_index(flat_tensor, batch_size: int) -> List[Tensor]:
    """
    Unrolls the flat tensor into list of tensors per batch item.
    As name suggest it undoes what flat_collate_tensors_with_batch_index does.
    :param flat_tensor:
    :param batch_size:
    :return: List of tensors
    """
    items = []
    batch_index_roi = [slice(None)] + [0] * (flat_tensor.ndim - 1)
    batch_index = flat_tensor[batch_index_roi]
    for i in range(batch_size):
        mask = batch_index == i
        items.append(flat_tensor[mask][..., 1:])
    return items
